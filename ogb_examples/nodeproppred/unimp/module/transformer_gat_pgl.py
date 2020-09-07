'''transformer_gcn
'''

import paddle.fluid as fluid
from pgl import graph_wrapper
from pgl.utils import paddle_helper
import math



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
            output = output / (hidden_size ** 0.5)
            return {"alpha": output, "v": src_feat["v_h"]}   # batch x h     batch x h x feat
        else:
            edge_feat = edge_feat["edge"]
            edge_feat = fluid.layers.reshape(edge_feat, [-1, num_heads, hidden_size])
            output = (src_feat["k_h"] + edge_feat) * dst_feat["q_h"]
            output = fluid.layers.reduce_sum(output, -1)
            output = output / (hidden_size ** 0.5)
            return {"alpha": output, "v": (src_feat["v_h"] + edge_feat)}   # batch x h     batch x h x feat

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
        h = fluid.layers.lod_reset(h, old_h)
        h = fluid.layers.sequence_pool(h, "sum")
        if concat:
            h = fluid.layers.reshape(h, [-1, num_heads * hidden_size])
        else:
            h = fluid.layers.reduce_mean(h, dim=1)
        return h
    
#     stdv = math.sqrt(6.0 / (feature.shape[-1] + hidden_size * num_heads))
#     q_w_attr=fluid.ParamAttr(initializer=fluid.initializer.UniformInitializer(low=-stdv, high=stdv))
    q_w_attr=fluid.ParamAttr(initializer=fluid.initializer.XavierInitializer())
    q_bias_attr=fluid.ParamAttr(initializer=fluid.initializer.ConstantInitializer(0.0))
    q = fluid.layers.fc(feature,
                         hidden_size * num_heads,
                       name=name + '_q_weight',
                       param_attr=q_w_attr,
                       bias_attr=q_bias_attr)
#     k_w_attr=fluid.ParamAttr(initializer=fluid.initializer.UniformInitializer(low=-stdv, high=stdv))
    k_w_attr=fluid.ParamAttr(initializer=fluid.initializer.XavierInitializer())
    k_bias_attr=fluid.ParamAttr(initializer=fluid.initializer.ConstantInitializer(0.0))
    k = fluid.layers.fc(feature,
                         hidden_size * num_heads,
                       name=name + '_k_weight',
                       param_attr=k_w_attr,
                       bias_attr=k_bias_attr)
#     v_w_attr=fluid.ParamAttr(initializer=fluid.initializer.UniformInitializer(low=-stdv, high=stdv))
    v_w_attr=fluid.ParamAttr(initializer=fluid.initializer.XavierInitializer())
    v_bias_attr=fluid.ParamAttr(initializer=fluid.initializer.ConstantInitializer(0.0))
    v = fluid.layers.fc(feature,
                         hidden_size * num_heads,
                       name=name + '_v_weight',
                       param_attr=v_w_attr,
                       bias_attr=v_bias_attr)
    
    reshape_q = fluid.layers.reshape(q, [-1, num_heads, hidden_size])
    reshape_k = fluid.layers.reshape(k, [-1, num_heads, hidden_size])
    reshape_v = fluid.layers.reshape(v, [-1, num_heads, hidden_size])

    msg = gw.send(
        send_attention,
        nfeat_list=[("q_h", reshape_q), ("k_h", reshape_k),
                    ("v_h", reshape_v)],
        efeat_list=edge_feature)
    output = gw.recv(msg, reduce_attention)

    return output
