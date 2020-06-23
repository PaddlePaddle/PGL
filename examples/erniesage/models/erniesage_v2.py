import pgl
import paddle.fluid as F
import paddle.fluid.layers as L
from models.base import BaseNet, BaseGNNModel
from models.ernie_model.ernie import ErnieModel


class ErnieSageV2(BaseNet):

    def build_inputs(self):
        inputs = super(ErnieSageV2, self).build_inputs()
        term_ids = L.data(
            "term_ids", shape=[None, self.config.max_seqlen], dtype="int64", append_batch_size=False)
        return inputs + [term_ids]

    def gnn_layer(self, gw, feature, hidden_size, act, initializer, learning_rate, name):
        def build_position_ids(src_ids, dst_ids):
            src_shape = L.shape(src_ids)
            src_batch = src_shape[0]
            src_seqlen = src_shape[1]
            dst_seqlen = src_seqlen - 1 # without cls

            src_position_ids = L.reshape(
                L.range(
                    0, src_seqlen, 1, dtype='int32'), [1, src_seqlen, 1],
                inplace=True) # [1, slot_seqlen, 1]
            src_position_ids = L.expand(src_position_ids, [src_batch, 1, 1]) # [B, slot_seqlen * num_b, 1]
            zero = L.fill_constant([1], dtype='int64', value=0)
            input_mask = L.cast(L.equal(src_ids, zero), "int32")  # assume pad id == 0 [B, slot_seqlen, 1]
            src_pad_len = L.reduce_sum(input_mask, 1, keep_dim=True) # [B, 1, 1]

            dst_position_ids = L.reshape(
                L.range(
                    src_seqlen, src_seqlen+dst_seqlen, 1, dtype='int32'), [1, dst_seqlen, 1],
                inplace=True) # [1, slot_seqlen, 1]
            dst_position_ids = L.expand(dst_position_ids, [src_batch, 1, 1]) # [B, slot_seqlen, 1]
            dst_position_ids = dst_position_ids - src_pad_len # [B, slot_seqlen, 1]

            position_ids = L.concat([src_position_ids, dst_position_ids], 1)
            position_ids = L.cast(position_ids, 'int64')
            position_ids.stop_gradient = True
            return position_ids


        def ernie_send(src_feat, dst_feat, edge_feat):
            """doc"""
            # input_ids
            cls = L.fill_constant_batch_size_like(src_feat["term_ids"], [-1, 1, 1], "int64", 1)
            src_ids = L.concat([cls, src_feat["term_ids"]], 1)
            dst_ids = dst_feat["term_ids"]

            # sent_ids
            sent_ids = L.concat([L.zeros_like(src_ids), L.ones_like(dst_ids)], 1)
            term_ids = L.concat([src_ids, dst_ids], 1)

            # position_ids
            position_ids = build_position_ids(src_ids, dst_ids)

            term_ids.stop_gradient = True
            sent_ids.stop_gradient = True
            ernie = ErnieModel(
                term_ids, sent_ids, position_ids,
                config=self.config.ernie_config)
            feature = ernie.get_pooled_output()
            return feature

        def erniesage_v2_aggregator(gw, feature, hidden_size, act, initializer, learning_rate, name):
            feature = L.unsqueeze(feature, [-1])
            msg = gw.send(ernie_send, nfeat_list=[("term_ids", feature)])
            neigh_feature = gw.recv(msg, lambda feat: F.layers.sequence_pool(feat, pool_type="sum"))

            term_ids = feature
            cls = L.fill_constant_batch_size_like(term_ids, [-1, 1, 1], "int64", 1)
            term_ids = L.concat([cls, term_ids], 1)
            term_ids.stop_gradient = True
            ernie = ErnieModel(
                term_ids, L.zeros_like(term_ids),
                config=self.config.ernie_config)
            self_feature = ernie.get_pooled_output()

            self_feature = L.fc(self_feature,
                                           hidden_size,
                                           act=act,
                                           param_attr=F.ParamAttr(name=name + "_l.w_0",
                                           learning_rate=learning_rate),
                                           bias_attr=name+"_l.b_0"
                                           )
            neigh_feature = L.fc(neigh_feature,
                                            hidden_size,
                                            act=act,
                                            param_attr=F.ParamAttr(name=name + "_r.w_0",
                                            learning_rate=learning_rate),
                                            bias_attr=name+"_r.b_0"
                                            )
            output = L.concat([self_feature, neigh_feature], axis=1)
            output = L.l2_normalize(output, axis=1)
            return output
        return erniesage_v2_aggregator(gw, feature, hidden_size, act, initializer, learning_rate, name)

    def gnn_layers(self, graph_wrappers, feature):
        features = [feature]

        initializer = None
        fc_lr = self.config.lr / 0.001

        for i in range(self.config.num_layers):
            if i == self.config.num_layers - 1:
                act = None
            else:
                act = "leaky_relu"

            feature = self.gnn_layer(
                graph_wrappers[i],
                feature,
                self.config.hidden_size,
                act,
                initializer,
                learning_rate=fc_lr,
                name="%s_%s" % ("erniesage_v2", i))
            features.append(feature)
        return features

    def __call__(self, graph_wrappers):
        inputs = self.build_inputs()
        feature = inputs[-1]
        features = self.gnn_layers(graph_wrappers, feature)
        outputs = [self.take_final_feature(features[-1], i, "final_fc") for i in inputs[:-1]]
        src_real_index = L.gather(graph_wrappers[0].node_feat['index'], inputs[0])
        outputs.append(src_real_index)
        return inputs, outputs


class ErnieSageModelV2(BaseGNNModel):
    def gen_net_fn(self, config):
        return ErnieSageV2(config)
