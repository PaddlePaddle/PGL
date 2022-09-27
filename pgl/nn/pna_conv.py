# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import math
import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from pgl.utils.logger import log
import pgl
__all__ = ['PNAConv']


class PNAConv(nn.Layer):
    """Implementation of Principal Neighbourhood Aggregation graph convolution operator

    This is an implementation of the paper Principal Neighbourhood 
    Aggregation for Graph Nets (https://arxiv.org/pdf/2004.05718).

    Args:
        input_size (int):the size of input.
        hidden_size (int): the size of output.
        aggregators (list): List of aggregation function keyword,
            choices in ["mean", "sum", "max", "min", "var", "std"]
        scalers: (list): List of scaler function keyword, 
            choices in ["identity", "amplification",
                        "attenuation", "linear", "inverse_linear"]
        deg (Tensor): Histogram of in-degrees of nodes in the training set for computed avg_deg for scalers
        towers (int, optional): Number of towers. Default: 1
        pre_layers (int, optional): Number of transformation layers before
            aggregation. Default: 1
        post_layers (int, optional): Number of transformation layers after
            aggregation. Default: 1
        divide_input (bool, optional): Whether the input features should
            be split between towers or not. Default: False
        use_edge (bool, optional): Whether to use edge feature. Default: False
    """

    def __init__(self,
                 input_size,
                 hidden_size,
                 aggregators,
                 scalers,
                 deg,
                 towers=1,
                 pre_layers=1,
                 post_layers=1,
                 divide_input=False,
                 use_edge=False):
        super(PNAConv, self).__init__()
        if divide_input:
            assert input_size % towers == 0
        assert hidden_size % towers == 0
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.aggregators = [AGGREGATOR[aggr] for aggr in aggregators]
        self.scalers = [SCALERS[scaler] for scaler in scalers]
        self.use_edge = use_edge
        self.towers = towers
        self.divide_input = divide_input
        self.pre_layers = pre_layers
        self.post_layers = post_layers
        self.F_in = input_size // towers if divide_input else input_size
        self.F_out = self.hidden_size // towers

        deg = deg.astype("float32")
        total_no_vertices = deg.sum()
        bin_degrees = paddle.arange(len(deg), dtype="float32")
        self.avg_deg = {
            'lin': ((bin_degrees * deg).sum() / total_no_vertices).item(),
            'log':
            (((bin_degrees + 1).log() * deg).sum() / total_no_vertices).item(),
            'exp': (
                (bin_degrees.exp() * deg).sum() / total_no_vertices).item(),
        }
        if use_edge:
            self.edge_mlp = paddle.nn.Linear(input_size, self.F_in)
        self.pre_nns = nn.LayerList()
        self.post_nns = nn.LayerList()
        for _ in range(self.towers):
            modules = [
                nn.Linear((3 if self.use_edge else 2) * self.F_in, self.F_in)
            ]
            for _ in range(self.pre_layers - 1):
                modules += [nn.ReLU()]
                modules += [nn.Linear(self.F_in, self.F_in)]
            self.pre_nns.append(nn.Sequential(*modules))
            input_size = (len(aggregators) * len(scalers) + 1) * self.F_in
            modules = [nn.Linear(input_size, self.F_out)]
            for _ in range(post_layers - 1):
                modules += [nn.ReLU()]
                modules += [nn.Linear(self.F_out, self.F_out)]
            self.post_nns.append(nn.Sequential(*modules))
        self.lin = nn.Linear(self.hidden_size, self.hidden_size)

    def _send_attention(self, src_feat, dst_feat, edge_feat):
        if "edge_feat" in edge_feat:
            edge_feat = self.edge_mlp(edge_feat['edge_feat'])
            edge_feat = edge_feat.reshape([-1, 1, self.F_in])
            edge_feat = edge_feat.tile([1, self.towers, 1])
            h = paddle.concat(
                [src_feat['h'], dst_feat['h'], edge_feat], axis=-1)
        else:
            h = paddle.concat([src_feat['h'], dst_feat['h']], axis=-1)
        hs = [nn(h[:, i]) for i, nn in enumerate(self.pre_nns)]
        return {"h": paddle.stack(hs, axis=1)}

    def _reduce_attention(self, msg):
        outs = [aggr(msg['h'], msg._segment_ids) for aggr in self.aggregators]
        out = paddle.concat(outs, axis=-1)
        return out.reshape([out.shape[0], -1])

    def forward(self, graph, feature, deg, edge_feat=None):
        """
        forward function of PNAConv
        Args:
            graph: pgl.graph instance.
            feature: A tensor with shape (num_nodes, input_size)
            deg: the in-degree of input nodes
            edge_feat(optional):  input edge features
        """
        if self.divide_input:
            feature = feature.reshape([-1, self.towers, self.F_in])
        else:
            feature = feature.reshape([-1, 1, self.F_in]).tile(
                [1, self.towers, 1])
        msg = graph.send(
            self._send_attention,
            src_feat={"h": feature},
            dst_feat={"h": feature},
            edge_feat={"edge_feat": edge_feat} if self.use_edge else {})

        out = graph.recv(reduce_func=self._reduce_attention, msg=msg)
        out = out.reshape([out.shape[0], self.towers, -1])
        deg = deg.astype("float32").reshape([-1, 1, 1])
        outs = [scaler(out, deg, self.avg_deg) for scaler in self.scalers]
        out = paddle.concat(outs, axis=-1)
        out = paddle.concat([feature, out], axis=-1)
        outs = [nn(out[:, i]) for i, nn in enumerate(self.post_nns)]
        out = paddle.concat(outs, axis=1)
        return self.lin(out)


def scale_identity(src, deg, avg_deg):
    """
    Implementation of identity scaler
    """
    return src


def scale_amplification(src, deg, avg_deg):
    """
    Implementation of amplification scaler
    """
    return src * (paddle.log(deg + 1) / avg_deg['log'])


def scale_attenuation(src, deg, avg_deg):
    """
    Implementation of attenuation scaler
    """
    scale = avg_deg['log'] / paddle.log(deg + 1)
    scale = paddle.where(deg == 0, paddle.ones(scale.shape), scale)
    return src * scale


def scale_linear(src, deg, avg_deg):
    """
    Implementation of linear scaler
    """
    return src * (deg / avg_deg['lin'])


def scale_inverse_linear(src, deg, avg_deg):
    """
    Implementation of inverse_linear scaler
    """

    scale = avg_deg['lin'] / deg
    scale = paddle.where(deg == 0, paddle.ones(scale.shape), scale)
    return src * scale


SCALERS = {
    'identity': scale_identity,
    'amplification': scale_amplification,
    'attenuation': scale_attenuation,
    'linear': scale_linear,
    'inverse_linear': scale_inverse_linear
}


def aggregate_sum(src, segment_id):
    """
    Implementation of sum aggregator
    """
    return pgl.math.segment_sum(src, segment_id)


def aggregate_mean(src, segment_id):
    """
    Implementation of mean aggregator
    """
    return pgl.math.segment_mean(src, segment_id)


def aggregate_max(src, segment_id):
    """
    Implementation of max aggregator
    """
    return pgl.math.segment_max(src, segment_id)


def aggregate_min(src, segment_id):
    """
    Implementation of min aggregator
    """
    return pgl.math.segment_min(src, segment_id)


def aggregate_var(src, segment_id):
    """
    Implementation of var aggregator
    """
    mean = aggregate_mean(src, segment_id)
    mean_squares = aggregate_mean(src * src, segment_id)
    return mean_squares - mean * mean


def aggregate_std(src, segment_id):
    """
    Implementation of std aggregator
    """
    return paddle.sqrt(
        paddle.nn.functional.relu(aggregate_var(src, segment_id)) + 1e-5)


AGGREGATOR = {
    "sum": aggregate_sum,
    "mean": aggregate_mean,
    "max": aggregate_max,
    "min": aggregate_min,
    "var": aggregate_var,
    "std": aggregate_std
}
