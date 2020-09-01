import numpy as np
import pgl
import paddle.fluid as F
import paddle.fluid.layers as L
from models.ernie_model.ernie import ErnieModel
from models.ernie_model.ernie import ErnieGraphModel
from models.ernie_model.ernie import ErnieConfig
from models import message_passing
from models.message_passing import copy_send


def get_layer(layer_type, gw, feature, hidden_size, act, initializer, learning_rate, name, is_test=False):
    return getattr(message_passing, layer_type)(gw, feature, hidden_size, act, initializer, learning_rate, name)


class Encoder(object):

    def __init__(self, config):
        self.config = config

    @classmethod
    def factory(cls, config):
        model_type = config.model_type
        if model_type == "ERNIESageV1":
            return ERNIESageV1Encoder(config)
        elif model_type == "ERNIESageV2":
            return ERNIESageV2Encoder(config)
        elif model_type == "ERNIESageV3":
            return ERNIESageV3Encoder(config)
        elif model_type == "ERNIESageV4":
            return ERNIESageV4Encoder(config)
        else:
            raise ValueError

    def __call__(self, graph_wrappers, inputs):
        raise NotImplementedError


class ERNIESageV1Encoder(Encoder):
    def __call__(self, graph_wrappers, inputs):
        feature = self.build_embedding(graph_wrappers[0].node_feat["term_ids"])

        initializer = None
        fc_lr = self.config.lr / 0.001
        for i in range(self.config.num_layers):
            if i == self.config.num_layers - 1:
                act = None
            else:
                act = "leaky_relu"
            feature = get_layer(
                self.config.layer_type,
                graph_wrappers[i],
                feature,
                self.config.hidden_size,
                act,
                initializer,
                learning_rate=fc_lr,
                name="%s_%s" % (self.config.layer_type, i))
    
        final_feats = [self.take_final_feature(feature, i, "final_fc") for i in inputs]
        return final_feats

    def build_embedding(self, term_ids):
        term_ids = L.unsqueeze(term_ids, [-1])
        ernie_config = self.config.ernie_config
        ernie = ErnieModel(
            src_ids=term_ids,
            sentence_ids=L.zeros_like(term_ids),
            task_ids=None,
            config=ernie_config,
            use_fp16=False,
            name="")
        feature = ernie.get_pooled_output()
        return feature

    def take_final_feature(self, feature, index, name):
        """take final feature"""
        feat = L.gather(feature, index, overwrite=False)

        if self.config.final_fc:
            feat = L.fc(feat,
                           self.config.hidden_size,
                           param_attr=F.ParamAttr(name=name + '_w'),
                           bias_attr=F.ParamAttr(name=name + '_b'))

        if self.config.final_l2_norm:
            feat = L.l2_normalize(feat, axis=1)
        return feat

class ERNIESageV2Encoder(Encoder):
    
    def __call__(self, graph_wrappers, inputs):
        feature = graph_wrappers[0].node_feat["term_ids"]
        feature = self.gnn_layer(graph_wrappers[0], feature, self.config.hidden_size, 'leaky_relu', None, 1., "erniesage_v2_0")

        initializer = None
        fc_lr = self.config.lr / 0.001
        for i in range(1, self.config.num_layers):
            if i == self.config.num_layers - 1:
                act = None
            else:
                act = "leaky_relu"
            feature = get_layer(
                self.config.layer_type,
                graph_wrappers[i],
                feature,
                self.config.hidden_size,
                act,
                initializer,
                learning_rate=fc_lr,
                name="%s_%s" % (self.config.layer_type, i))
    
        final_feats = [self.take_final_feature(feature, i, "final_fc") for i in inputs]
        return final_feats

    def take_final_feature(self, feature, index, name):
        """take final feature"""
        feat = L.gather(feature, index, overwrite=False)

        if self.config.final_fc:
            feat = L.fc(feat,
                           self.config.hidden_size,
                           param_attr=F.ParamAttr(name=name + '_w'),
                           bias_attr=F.ParamAttr(name=name + '_b'))

        if self.config.final_l2_norm:
            feat = L.l2_normalize(feat, axis=1)
        return feat

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

class ERNIESageV3Encoder(Encoder):

    def __call__(self, graph_wrappers, inputs):
        feature = graph_wrappers[0].node_feat["term_ids"]
        feature = self.gnn_layer(graph_wrappers[0], feature, self.config.hidden_size, 'leaky_relu', None, 1., "erniesage_v3_0")

        final_feats = [self.take_final_feature(feature, i, "final_fc") for i in inputs]
        return final_feats

    def gnn_layer(self, gw, feature, hidden_size, act, initializer, learning_rate, name):
        def ernie_recv(feat):
            """doc"""
            num_neighbor = self.config.samples[0]
            pad_value = L.zeros([1], "int64")
            out, _ = L.sequence_pad(feat, pad_value=pad_value, maxlen=num_neighbor)
            out = L.reshape(out, [0, self.config.max_seqlen*num_neighbor])
            return out

        def erniesage_v3_aggregator(gw, feature, hidden_size, act, initializer, learning_rate, name):
            msg = gw.send(copy_send, nfeat_list=[("h", feature)])
            neigh_feature = gw.recv(msg, ernie_recv)
            neigh_feature = L.cast(L.unsqueeze(neigh_feature, [-1]), "int64")

            feature = L.unsqueeze(feature, [-1])
            cls = L.fill_constant_batch_size_like(feature, [-1, 1, 1], "int64", 1)
            term_ids = L.concat([cls, feature[:, :-1], neigh_feature], 1)
            term_ids.stop_gradient = True
            return term_ids
        return erniesage_v3_aggregator(gw, feature, hidden_size, act, initializer, learning_rate, name)

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
                name="%s_%s" % ("erniesage_v3", i))
            features.append(feature)
        return features

    def take_final_feature(self, feature, index, name):
        """take final feature"""
        feat = L.gather(feature, index, overwrite=False)

        ernie_config = self.config.ernie_config
        ernie = ErnieGraphModel(
            src_ids=feat,
            config=ernie_config,
            slot_seqlen=self.config.max_seqlen)
        feat = ernie.get_pooled_output()
        fc_lr = self.config.lr / 0.001

        if self.config.final_fc:
            feat = L.fc(feat,
                           self.config.hidden_size,
                           param_attr=F.ParamAttr(name=name + '_w'),
                           bias_attr=F.ParamAttr(name=name + '_b'))

        if self.config.final_l2_norm:
            feat = L.l2_normalize(feat, axis=1)
        return feat
