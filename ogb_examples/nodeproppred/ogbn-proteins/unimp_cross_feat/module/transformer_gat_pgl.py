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
'''transformer_gcn
'''

import paddle.fluid as fluid
import paddle.fluid.layers as L
from pgl import graph_wrapper
from pgl.utils import paddle_helper
import math


def get_degree(edge, num_nodes):
    init_output = L.fill_constant(shape=[num_nodes], value=0, dtype="float32")
    init_output.stop_gradient = True
    final_output = L.scatter(
        init_output,
        edge,
        L.full_like(
            edge, 1, dtype="float32"),
        overwrite=False)
    return final_output


class PartitionWrapper(graph_wrapper.BaseGraphWrapper):
    """Implement of Partition Wrapper"""

    def __init__(self, graph_wrapper, part_id, num_parts):
        super(PartitionWrapper, self).__init__()

        # Copy Node's information
        for key, value in graph_wrapper.node_feat.items():
            self.node_feat_tensor_dict[key] = value

        self._num_nodes = graph_wrapper.num_nodes
        self._graph_lod = graph_wrapper.graph_lod
        self._num_graph = graph_wrapper.num_graph

        # Dropout Edges
        src, dst = graph_wrapper.edges
        keeped = L.cast((dst % num_parts) == part_id, dtype="float32")

        keeped = (keeped > 0.5)
        self.keeped = keeped
        self._num_edges = L.reduce_sum(L.cast(keeped, "int32"))
        #L.Print(self._num_edges, message="Part-%s num edges" % part_id)
        src = paddle_helper.masked_select(src, keeped)
        dst = paddle_helper.masked_select(dst, keeped)
        src.stop_gradient = True
        dst.step_gradient = True
        self._edges_src = src
        self._edges_dst = dst

        for key, value in graph_wrapper.edge_feat.items():
            self.edge_feat_tensor_dict[key] = paddle_helper.masked_select(
                value, keeped)

        self._edge_uniq_dst, _, uniq_count = L.unique_with_counts(
            dst, dtype="int32")
        self._edge_uniq_dst.stop_gradient = True
        last = L.reduce_sum(uniq_count, keep_dim=True)
        uniq_count = L.cumsum(uniq_count, exclusive=True)
        self._edge_uniq_dst_count = L.concat([uniq_count, last])
        self._edge_uniq_dst_count.stop_gradient = True
        self._indegree = get_degree(self._edges_dst, self._num_nodes)


def split_gw(gw, partition):
    gw_list = []
    for i in range(partition):
        gw_list.append(PartitionWrapper(gw, i, partition))
    return gw_list


def transformer_gat_pgl(gw,
                        feature,
                        hidden_size,
                        name,
                        num_heads=4,
                        attn_drop=0,
                        edge_feature=None,
                        concat=True,
                        is_test=False):
    '''transformer_gat_pgl
    '''

    def send_attention(src_feat, dst_feat, edge_feat):
        if edge_feat is None or not edge_feat:
            output = src_feat["k_h"] * dst_feat["q_h"]
            output = fluid.layers.reduce_sum(output, -1)
            return {
                "alpha": output,
                "v": src_feat["v_h"]
            }  # batch x h     batch x h x feat
        else:
            edge_feat = edge_feat["edge"]
            edge_feat = fluid.layers.reshape(edge_feat,
                                             [-1, num_heads, hidden_size])
            output = (src_feat["k_h"] + edge_feat) * dst_feat["q_h"]
            output = fluid.layers.reduce_sum(output, -1)
            return {
                "alpha": output,
                "v": (src_feat["v_h"] + edge_feat)
            }  # batch x h     batch x h x feat

    def reduce_attention(msg):
        alpha = msg["alpha"]  # lod-tensor (batch_size, seq_len, num_heads)
        h = msg["v"]
        alpha = paddle_helper.sequence_softmax(alpha)
        old_h = h

        if attn_drop > 1e-15:
            alpha = fluid.layers.dropout(
                alpha,
                dropout_prob=attn_drop,
                is_test=is_test,
                dropout_implementation="upscale_in_train")
        h = h * alpha
        #h = fluid.layers.lod_reset(h, old_h)
        h = fluid.layers.sequence_pool(h, "sum")
        if concat:
            h = fluid.layers.reshape(h, [-1, num_heads * hidden_size])
        else:
            h = fluid.layers.reduce_mean(h, dim=1)
        return h

    q_w_attr = fluid.ParamAttr(
        initializer=fluid.initializer.XavierInitializer())
    q_bias_attr = fluid.ParamAttr(
        initializer=fluid.initializer.ConstantInitializer(0.0))
    q = fluid.layers.fc(feature,
                        hidden_size * num_heads,
                        name=name + '_q_weight',
                        param_attr=q_w_attr,
                        bias_attr=q_bias_attr)
    q = q / (hidden_size**0.5)

    k_w_attr = fluid.ParamAttr(
        initializer=fluid.initializer.XavierInitializer())
    k_bias_attr = fluid.ParamAttr(
        initializer=fluid.initializer.ConstantInitializer(0.0))
    k = fluid.layers.fc(feature,
                        hidden_size * num_heads,
                        name=name + '_k_weight',
                        param_attr=k_w_attr,
                        bias_attr=k_bias_attr)

    v_w_attr = fluid.ParamAttr(
        initializer=fluid.initializer.XavierInitializer())
    v_bias_attr = fluid.ParamAttr(
        initializer=fluid.initializer.ConstantInitializer(0.0))
    v = fluid.layers.fc(feature,
                        hidden_size * num_heads,
                        name=name + '_v_weight',
                        param_attr=v_w_attr,
                        bias_attr=v_bias_attr)

    reshape_q = fluid.layers.reshape(q, [-1, num_heads, hidden_size])
    reshape_k = fluid.layers.reshape(k, [-1, num_heads, hidden_size])
    reshape_v = fluid.layers.reshape(v, [-1, num_heads, hidden_size])

    if not isinstance(gw, list):
        msg = gw.send(
            send_attention,
            nfeat_list=[("q_h", reshape_q), ("k_h", reshape_k),
                        ("v_h", reshape_v)],
            efeat_list=edge_feature)
        output = gw.recv(msg, reduce_attention)
        return output
    else:
        checkpoints = []
        outputs = []
        for batch_no, (batch_gw,
                       batch_edge_feat) in enumerate(zip(gw, edge_feature)):
            msg = batch_gw.send(
                send_attention,
                nfeat_list=[("q_h", reshape_q), ("k_h", reshape_k),
                            ("v_h", reshape_v)],
                efeat_list=batch_edge_feat)
            output = batch_gw.recv(msg, reduce_attention)
            outputs.append(output)
        outputs = L.sum(outputs)
        return outputs, checkpoints
