import pgl
import paddle.fluid as F
import paddle.fluid.layers as L
from pgl.utils import paddle_helper
from pgl import message_passing
import math

def graph_transformer(name, gw,
        feature,
        hidden_size,
        num_heads=4,
        attn_drop=False,
        edge_feature=None,
        concat=True,
        skip_feat=True,
        gate=False,
        layer_norm=True, 
        relu=True, 
        is_test=False):
    """Implementation of graph Transformer from UniMP

    This is an implementation of the paper Unified Massage Passing Model for Semi-Supervised Classification
    (https://arxiv.org/abs/2009.03509).

    Args:
        name: Granph Transformer layer names.
        
        gw: Graph wrapper object (:code:`StaticGraphWrapper` or :code:`GraphWrapper`)

        feature: A tensor with shape (num_nodes, feature_size).

        hidden_size: The hidden size for graph transformer.

        num_heads: The head number in graph transformer.

        attn_drop: Dropout rate for attention.
        
        edge_feature: A tensor with shape (num_edges, feature_size).
        
        concat: Reshape the output (num_nodes, num_heads, hidden_size) by concat (num_nodes, hidden_size * num_heads) or mean (num_nodes, hidden_size)
        
        skip_feat: Whether use skip connect
        
        gate: Whether add skip_feat and output up with gate weight
        
        layer_norm: Whether use layer_norm for output
        
        relu: Whether use relu activation for output

        is_test: Whether in test phrase.

    Return:
        A tensor with shape (num_nodes, hidden_size * num_heads) or (num_nodes, hidden_size)
    """
    def send_attention(src_feat, dst_feat, edge_feat):
        if edge_feat is None or not edge_feat:
            output = src_feat["k_h"] * dst_feat["q_h"]
            output = L.reduce_sum(output, -1)
            output = output / (hidden_size ** 0.5)
#             alpha = paddle_helper.sequence_softmax(output)
            return {"alpha": output, "v": src_feat["v_h"]}   # batch x h     batch x h x feat
        else:
            edge_feat = edge_feat["edge"]
            edge_feat = L.reshape(edge_feat, [-1, num_heads, hidden_size])
            output = (src_feat["k_h"] + edge_feat) * dst_feat["q_h"]
            output = L.reduce_sum(output, -1)
            output = output / (hidden_size ** 0.5)
#             alpha = paddle_helper.sequence_softmax(output)
            return {"alpha": output, "v": (src_feat["v_h"] + edge_feat)}   # batch x h     batch x h x feat

    class Reduce_attention():
        def __init__(self,):
            self.alpha = None
        def __call__(self, msg):
            alpha = msg["alpha"]  # lod-tensor (batch_size, num_heads)
            if attn_drop:
                old_h = alpha
                dropout = F.data(name='attn_drop', shape=[1], dtype="int64")
                u = L.uniform_random(shape=L.cast(L.shape(alpha)[:1], 'int64'), min=0., max=1.)
                keeped = L.cast(u > dropout, dtype="float32")
                self_attn_mask = L.scale(x=keeped, scale=10000.0, bias=-1.0, bias_after_scale=False)
                n_head_self_attn_mask = L.stack( x=[self_attn_mask] * num_heads, axis=1)
                n_head_self_attn_mask.stop_gradient = True
                alpha = n_head_self_attn_mask+ alpha
                alpha = L.lod_reset(alpha, old_h)

            h = msg["v"]
            alpha = paddle_helper.sequence_softmax(alpha)
            
            self.alpha = alpha
            old_h = h
            h_mean = L.sequence_pool(h, "average")
            h = h * alpha
            h = L.lod_reset(h, old_h)
            h = L.sequence_pool(h, "sum")

            h = h * 0.8 + h_mean * 0.2
            
            if concat:
                h = L.reshape(h, [-1, num_heads * hidden_size])
            else:
                h = L.reduce_mean(h, dim=1)
            return h
    reduce_attention = Reduce_attention()
    
    q = linear(feature, hidden_size * num_heads, name=name + '_q_weight', init_type='gcn')
    k = linear(feature, hidden_size * num_heads, name=name + '_k_weight', init_type='gcn')
    v = linear(feature, hidden_size * num_heads, name=name + '_v_weight', init_type='gcn')
    
    
    reshape_q = L.reshape(q, [-1, num_heads, hidden_size])
    reshape_k = L.reshape(k, [-1, num_heads, hidden_size])
    reshape_v = L.reshape(v, [-1, num_heads, hidden_size])

    msg = gw.send(
        send_attention,
        nfeat_list=[("q_h", reshape_q), ("k_h", reshape_k),
                    ("v_h", reshape_v)],
        efeat_list=edge_feature)
    out_feat = gw.recv(msg, reduce_attention)
    checkpoints=[out_feat]
    
    if skip_feat:
        if concat:

            out_feat, cks = appnp(gw, out_feat, k_hop=3, name=name+"_appnp")
#             out_feat, cks = appnp(gw, out_feat, k_hop=3)
            checkpoints.append(out_feat)
    
#             The UniMP-xxlarge will come soon.
#             out_feat, cks = appnp(gw, out_feat, k_hop=6)
#             out_feat, cks = appnp(gw, out_feat, k_hop=9)
#             checkpoints = checkpoints + cks

            
            skip_feature = linear(feature, hidden_size * num_heads, name=name + '_skip_weight', init_type='lin')
        else:
            
            skip_feature = linear(feature, hidden_size, name=name + '_skip_weight', init_type='lin')
            
        if gate:
            temp_output = L.concat([skip_feature, out_feat, out_feat - skip_feature], axis=-1)
            gate_f = L.sigmoid(linear(temp_output, 1, name=name + '_gate_weight', init_type='lin'))
            out_feat = skip_feature * gate_f + out_feat * (1 - gate_f)
        else:
            out_feat = skip_feature + out_feat
            
    if layer_norm:
        lay_norm_attr = F.ParamAttr(initializer=F.initializer.ConstantInitializer(value=1))
        lay_norm_bias = F.ParamAttr(initializer=F.initializer.ConstantInitializer(value=0))
        out_feat = L.layer_norm(out_feat, name=name + '_layer_norm', 
                                  param_attr=lay_norm_attr, 
                                  bias_attr=lay_norm_bias,
                                  scale=False,
                                  shift=False)
    if relu:
        out_feat = L.relu(out_feat)
    
    return out_feat, reduce_attention.alpha, checkpoints


def appnp(gw, feature, alpha=0.2, k_hop=10, name=""):
    """Implementation of APPNP of "Predict then Propagate: Graph Neural Networks
    meet Personalized PageRank"  (ICLR 2019). 
    Args:
        gw: Graph wrapper object (:code:`StaticGraphWrapper` or :code:`GraphWrapper`)
        feature: A tensor with shape (num_nodes, feature_size).
        edge_dropout: Edge dropout rate.
        k_hop: K Steps for Propagation
    Return:
        A tensor with shape (num_nodes, hidden_size)
    """

    def send_src_copy(src_feat, dst_feat, edge_feat):
        feature = src_feat["h"]
        return feature
    
    def get_norm(indegree):
        float_degree = L.cast(indegree, dtype="float32")
        float_degree = L.clamp(float_degree, min=1.0)
        norm = L.pow(float_degree, factor=-0.5) 
        return norm
    
    cks = []
    h0 = feature
    ngw = gw 
    norm = get_norm(ngw.indegree())
    
    for i in range(k_hop):
            
        feature = feature * norm
        msg = gw.send(send_src_copy, nfeat_list=[("h", feature)])
        feature = gw.recv(msg, "sum")
        feature = feature * norm
        #feature = feature * (1 - alpha) + h0 * alpha

        fan_in = feature.shape[-1]*3
        bias_bound = 1.0 / math.sqrt(fan_in)
        fc_bias_attr = F.ParamAttr(initializer=F.initializer.UniformInitializer(low=-bias_bound, high=bias_bound))

        negative_slope = math.sqrt(5)
        gain = math.sqrt(2.0 / (1 + negative_slope ** 2))
        std = gain / math.sqrt(fan_in)
        weight_bound = math.sqrt(3.0) * std
        fc_w_attr = F.ParamAttr(initializer=F.initializer.UniformInitializer(low=-weight_bound, high=weight_bound))
        
        gate_f = L.fc([feature, h0, feature - h0],
                     1,
                       param_attr=fc_w_attr,
                       name=name + 'appnp_gate_' + str(i),
                       bias_attr=fc_bias_attr)
        
        alpha = L.sigmoid(gate_f)
        feature = feature * (1 - alpha) + h0 * alpha
        
        if (i+1) % 3 == 0:
            cks.append(feature)
    return feature, cks

def attn_appnp(gw, feature, attn, alpha=0.2, k_hop=10):
    """Attention based APPNP to Make model output deeper
    Args:
        gw: Graph wrapper object (:code:`StaticGraphWrapper` or :code:`GraphWrapper`)
        attn: Using the attntion as transition matrix for APPNP
        feature: A tensor with shape (num_nodes, feature_size).
        k_hop: K Steps for Propagation
    Return:
        A tensor with shape (num_nodes, hidden_size)
    """
    def send_src_copy(src_feat, dst_feat, edge_feat):
        feature = src_feat["h"]
        return feature

    h0 = feature
    attn = L.reduce_mean(attn, 1)
    for i in range(k_hop):
        msg = gw.send(send_src_copy, nfeat_list=[("h", feature)])
        msg = msg * attn
        feature = gw.recv(msg, "sum")
        feature = feature * (1 - alpha) + h0 * alpha
    return feature

def linear(input, hidden_size, name, with_bias=True, init_type='gcn'):
    """fluid.layers.fc with different init_type
    """
    
    if init_type == 'gcn':
        fc_w_attr = F.ParamAttr(initializer=F.initializer.XavierInitializer())
        fc_bias_attr = F.ParamAttr(initializer=F.initializer.ConstantInitializer(0.0))
    else:
        fan_in = input.shape[-1]
        bias_bound = 1.0 / math.sqrt(fan_in)
        fc_bias_attr = F.ParamAttr(initializer=F.initializer.UniformInitializer(low=-bias_bound, high=bias_bound))

        negative_slope = math.sqrt(5)
        gain = math.sqrt(2.0 / (1 + negative_slope ** 2))
        std = gain / math.sqrt(fan_in)
        weight_bound = math.sqrt(3.0) * std
        fc_w_attr = F.ParamAttr(initializer=F.initializer.UniformInitializer(low=-weight_bound, high=weight_bound))
    
    if not with_bias:
        fc_bias_attr = False
        
    output = L.fc(input,
        hidden_size,
        param_attr=fc_w_attr,
        name=name,
        bias_attr=fc_bias_attr)
    return output
