# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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


import paddle.fluid as fluid
import paddle.fluid.layers as L
import paddle.fluid.layers as layers
from pgl.utils import paddle_helper
import pgl
    

def masked_select(input, mask):
    """masked_select
    
    Slice the value from given Mask
   
    Args:
        input: Input tensor to be selected
         
        mask: A bool tensor for sliced.
  
    Return:
        Part of inputs where mask is True. 
    """
    index = L.where(mask)
    return L.gather(input, index, overwrite=False)


class BigBirdWrapper(pgl.graph_wrapper.BaseGraphWrapper):
    """Implement of Big Bird by PGL graph wrapper """
    def __init__(self, input_mask):
        super(BigBirdWrapper, self).__init__()
        max_seqlen = L.shape(input_mask)[1]
        input_mask = L.reshape(input_mask, [-1])
        num_nodes = L.shape(input_mask)[0]
        src, dst = build_edges(num_nodes, input_mask, max_seqlen)
        self._edges_src = src
        self._edges_dst = dst 
        self._edges_src.stop_gradient=True
        self._edges_dst.stop_gradient=True
        self._num_nodes = num_nodes
        self._num_edges = L.shape(self._edges_src)[0]
        self._node_ids = L.range(0, self._num_nodes, step=1, dtype="int32")
        self._edge_uniq_dst, _, uniq_count = L.unique_with_counts(self._edges_dst, dtype="int32")
        self._edge_uniq_dst.stop_gradient=True
        last = L.reduce_sum(uniq_count, keep_dim=True)
        uniq_count = L.cumsum(uniq_count, exclusive=True)
        self._edge_uniq_dst_count = L.concat([uniq_count, last])
        self._edge_uniq_dst_count.stop_gradient=True


def select_edges(src, dst, input_mask, num_nodes, max_seqlen):
    src = fluid.layers.elementwise_max(src, num_nodes * 0)
    dst = fluid.layers.elementwise_max(dst, num_nodes * 0)
    src = fluid.layers.elementwise_min(src, num_nodes - 1)
    dst = fluid.layers.elementwise_min(dst, num_nodes - 1)

    conditions = []
    conditions.append(L.gather(input_mask, src) > 0.5)
    conditions.append(L.gather(input_mask, dst) > 0.5)
    block_src = src / max_seqlen
    block_dst = dst / max_seqlen
    conditions.append(block_src == block_dst)
    mask = None
    for cond in conditions:
        if mask is None:
            mask = cond
        else:
            mask = L.logical_and(mask, cond)
    
    dst = masked_select(dst, mask)
    src = masked_select(src, mask)
    return src, dst


def uniq_edges(src, dst, num_nodes):
    sorted_dst = L.cast(dst, dtype="int64")
    sorted_src = L.cast(src, dtype="int64")
    num_nodes = L.cast(num_nodes, dtype="int64")
    edge_hash = sorted_dst * num_nodes + sorted_src
    edge_hash, _  = L.argsort(edge_hash)
    edge_hash, _ = L.unique(edge_hash, dtype="int64")
    sorted_src = L.elementwise_mod(edge_hash, num_nodes)
    sorted_dst = L.elementwise_div(edge_hash, num_nodes)
    sorted_src = L.cast(sorted_src, dtype="int32")
    sorted_dst = L.cast(sorted_dst, dtype="int32")
    return sorted_src, sorted_dst
    

def build_edges(num_nodes, input_mask, max_seqlen):
    edges = L.range(start=0, end=num_nodes, step=1, dtype="int32")
    all_edges = []
    # Window
    filter_func = lambda x, y: select_edges(x, y, input_mask, num_nodes, max_seqlen)

    all_edges.append(filter_func(edges - 1, edges)) # win-1
    all_edges.append(filter_func(edges + 1, edges)) # win-2
    all_edges.append(filter_func(edges, edges)) #self-loop

    # Global Assume [CLS] is the first token.

    # vertical cls-window attention 
    cls_position = edges / max_seqlen * max_seqlen
    all_edges.append(filter_func(cls_position, edges))

    # horizontal cls attention
    all_edges.append(filter_func(edges, cls_position))

    # Random
    for i in range(2):
        rand_edge = L.floor(L.uniform_random(min=0, max=1, shape=[num_nodes]) * L.cast(max_seqlen, dtype="float32"))
        rand_edge = L.cast(rand_edge, dtype="int32") + cls_position
        all_edges.append(filter_func(rand_edge, edges))

    if len(all_edges) > 1:
        src = L.concat([ s for s, d in all_edges], 0)
        dst = L.concat([ d for s, d in all_edges], 0)
    else:
        src = all_edges[0][0]
        dst = all_edges[0][1]

    # sort edges 
    sorted_src, sorted_dst = uniq_edges(src, dst, num_nodes)
    return sorted_src, sorted_dst


def sparse_scaled_dot_product_attention(q, k, v, input_mask, dropout_rate, n_head, d_key, d_value):
    def send_q_k_spmm(src_feat, dst_feat, edge_feat):
        # q [ num_edges, n_head * dim]
        # k [ num_edges, n_head * dim]
        # v [ num_edges, n_head * dim]
        _q = dst_feat["q"] 
        _k = src_feat["k"] 
        _v = src_feat["v"] 
        _q = L.reshape(_q, [-1, n_head, _q.shape[-1] // n_head])
        _k = L.reshape(_k, [-1, n_head, _k.shape[-1] // n_head])
        score = L.reduce_sum(_q * _k, -1) # [num_edge, n_head]
        return { "score": score, "value": _v}

    def recv_score_v_spmm(msg):
        score = msg["score"]
        score = paddle_helper.sequence_softmax(score)
        score = layers.dropout(
                score,
                dropout_prob=dropout_rate,
                dropout_implementation="upscale_in_train",
                is_test=False)

        score = L.reshape(score, [-1, n_head, 1])
        _v = msg["value"]
        _new_v = L.reshape(_v, [-1, n_head, _v.shape[-1] // n_head])

        _new_v = _new_v * score

        _new_v = L.reshape(_new_v, [-1, _v.shape[-1]])
        _new_v = L.lod_reset(_new_v, _v)
        return L.sequence_pool(_new_v, "sum")

    graph_wrapper = BigBirdWrapper(input_mask)
    old_v = v

    q =  L.reshape(q, [-1, d_key * n_head])
    k =  L.reshape(k, [-1, d_key * n_head])
    v =  L.reshape(v, [-1, d_value * n_head])

    q = L.scale(q, scale=d_key ** -0.5)
    msg = graph_wrapper.send(send_q_k_spmm, nfeat_list=[("k", k), ("v", v), ("q", q)])
    out = graph_wrapper.recv(msg, recv_score_v_spmm)
    out = L.reshape(out, [-1, L.shape(old_v)[1], d_value * n_head])
    return out, out


