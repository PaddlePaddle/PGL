# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
# Copyright (c) 2021, rusty1s(github).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
    Base GNN module for GNNAutoScale, partially refers to `pygas`(https://github.com/rusty1s/pyg_autoscale).
"""

import os
import sys

import numpy as np
import paddle
from pgl.utils.stream_pool import StreamPool, async_write

sys.path.insert(0, os.path.abspath(".."))
from history import History
from utils import process_batch_data, check_device


class ScalableGNN(paddle.nn.Layer):
    def __init__(self,
                 num_nodes,
                 num_layers,
                 hidden_dim,
                 pool_size=None,
                 buffer_size=None):
        super().__init__()

        self.num_nodes = num_nodes
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.pool_size = num_layers - 1 if pool_size is None else pool_size
        self.buffer_size = buffer_size
        self.pool = None
        self._async = False
        self._fout = None

        self.histories = paddle.nn.LayerList(
            [History(num_nodes, hidden_dim) for _ in range(num_layers - 1)])

        self._init_pool()

    def _init_pool(self):
        if (self.check_emb_on_pin_memory() and self.pool_size is not None and
                self.buffer_size is not None and len(self.histories) > 0):
            self.pool = StreamPool(self.pool_size, self.buffer_size,
                                   self.hidden_dim)

    def check_emb_on_pin_memory(self):
        if len(self.histories) > 0:
            place = self.histories[0].emb.place
            if str(place).startswith("CUDAPinned"):
                return True
        return False

    def __call__(self,
                 subgraph,
                 x,
                 norm=None,
                 batch_size=None,
                 n_id=None,
                 offset=None,
                 count=None,
                 loader=None,
                 **kwargs):
        # TODO(daisiming): Add forward_pre_hook like paddle.nn.Layer, etc.

        if loader is not None:
            return self.inference(loader, x, norm)

        self._async = (self.pool is not None and batch_size is not None and
                       n_id is not None and offset is not None and
                       count is not None)

        if self._async and batch_size < len(n_id):
            for history in self.histories:
                cpu_nid = n_id[batch_size:].cpu()
                empty = paddle.to_tensor(np.array([]), place=paddle.CPUPlace())
                self.pool.async_pull(history.emb, cpu_nid, empty, empty)

        out = self.forward(subgraph, x, norm, batch_size, n_id, offset, count,
                           **kwargs)

        if self._async and batch_size < len(n_id):
            for history in self.histories:
                self.pool.sync_push()

        self._async = False

        return out

    def push_and_pull(self,
                      history,
                      x,
                      batch_size=None,
                      n_id=None,
                      offset=None,
                      count=None):

        if len(self.histories) == 0:
            return x

        if n_id is None:
            return x

        assert n_id is not None
        assert batch_size is not None, "batch_size should not be None"

        if batch_size == len(n_id):
            if self._async:
                self.pool.async_push(x[:batch_size], history.emb, offset,
                                     count)
            return x

        if self._async:
            out_batch_size = len(n_id) - batch_size
            out = self.pool.sync_pull()[:out_batch_size]
            self.pool.async_push(x[:batch_size], history.emb, offset, count)
            out = paddle.concat([x[:batch_size], out], axis=0)
            self.pool.free_pull()
            return out
        else:
            # TODO(daisiming): Due to some limitations of Paddle OP, we leave here as a todo.
            return x

    @paddle.no_grad()
    def _final_out(self):
        if self._fout is None:
            self._fout = paddle.to_tensor(
                np.zeros((self.num_nodes, self.output_size)),
                place=paddle.CUDAPinnedPlace(),
                dtype="float32")
        return self._fout

    @paddle.no_grad()
    def inference(self, loader, feature, norm):
        # We add state: {} here, which is used to store additional state, 
        # such as residual connections.
        loader = [(sub_data, {}) for sub_data in loader]

        if len(self.histories) == 0:
            for sub_data, state in loader:
                g, batch_size, n_id, offset, count, feat, sub_norm = \
                    process_batch_data(sub_data, feature, norm)
                out = self.forward_layer(0, g, feat, sub_norm,
                                         state)[:batch_size]
                # Push out to self._final_out()
                async_write(out, self._final_out(), offset, count)
            return self._final_out()

        # Push outputs of 0-th layer to the history.
        for sub_data, state in loader:
            g, batch_size, n_id, offset, count, feat, sub_norm = \
                process_batch_data(sub_data, feature, norm)
            out = self.forward_layer(0, g, feat, sub_norm, state)[:batch_size]
            self.pool.async_push(out, self.histories[0].emb, offset, count)
        self.pool.sync_push()

        for i in range(1, len(self.histories)):
            # Pull the complete layer-wise history.
            for sub_data, _ in loader:
                _, batch_size, n_id, offset, count, _, _ = \
                    process_batch_data(sub_data)
                if batch_size < len(n_id):
                    cpu_nid = n_id[batch_size:].cpu()
                    self.pool.async_pull(self.histories[i - 1].emb, cpu_nid,
                                         offset, count)
            for sub_data, state in loader:
                g, batch_size, n_id, offset, count, _, sub_norm = \
                    process_batch_data(sub_data, norm=norm)
                x = self.pool.sync_pull()[:len(n_id)]
                out = self.forward_layer(i, g, x, sub_norm, state)[:batch_size]
                self.pool.async_push(out, self.histories[i].emb, offset, count)
                self.pool.free_pull()
            self.pool.sync_push()

        # Pull the histories from last layer.
        for sub_data, _ in loader:
            _, batch_size, n_id, offset, count, _, _ = \
                process_batch_data(sub_data)
            if batch_size < len(n_id):
                cpu_nid = n_id[batch_size:].cpu()
                self.pool.async_pull(self.histories[-1].emb, cpu_nid, offset,
                                     count)

        # Compute final embeddings.
        for sub_data, state in loader:
            g, batch_size, n_id, offset, count, _, sub_norm = \
                process_batch_data(sub_data, norm=norm)
            x = self.pool.sync_pull()[:len(n_id)]
            out = self.forward_layer(self.num_layers - 1, g, x, sub_norm,
                                     state)[:batch_size]
            self.pool.async_push(out, self._final_out(), offset, count)
            self.pool.free_pull()
        self.pool.sync_push()

        return self._final_out()

    @paddle.no_grad()
    def forward_layer(self, layer, graph, x, norm=None, state=None):
        raise NotImplementedError
