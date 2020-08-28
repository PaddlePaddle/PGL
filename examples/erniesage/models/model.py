import numpy as np
import pgl
import paddle.fluid as F
import paddle.fluid.layers as L
from models.encoder import Encoder
from models.loss import Loss

class BaseModel(object):

    def __init__(self, config):
        self.config = config
        datas, graph_wrappers, loss, outputs = self.forward()
        self.build(datas, graph_wrappers, loss, outputs)

    def forward(self):
        raise NotImplementedError

    def build(self, datas, graph_wrappers, loss, outputs):
        self.datas = datas
        self.graph_wrappers = graph_wrappers
        self.loss = loss
        self.outputs = outputs
        self.build_feed_list()
        self.build_data_loader()

    def build_feed_list(self):
        self.feed_list = []
        for i in range(len(self.graph_wrappers)):
            self.feed_list.extend(self.graph_wrappers[i].holder_list)
        self.feed_list.extend(self.datas)

    def build_data_loader(self):
        self.data_loader = F.io.PyReader(
            feed_list=self.feed_list, capacity=20, use_double_buffer=True, iterable=True)


class LinkPredictModel(BaseModel):

    def forward(self):
        # datas
        user_index = L.data(
            "user_index", shape=[None], dtype="int64", append_batch_size=False)
        pos_item_index = L.data(
            "pos_item_index", shape=[None], dtype="int64", append_batch_size=False)
        neg_item_index = L.data(
            "neg_item_index", shape=[None], dtype="int64", append_batch_size=False)
        user_real_index = L.data(
            "user_real_index", shape=[None], dtype="int64", append_batch_size=False)
        pos_item_real_index = L.data(
            "pos_item_real_index", shape=[None], dtype="int64", append_batch_size=False)
        datas = [user_index, pos_item_index, neg_item_index, user_real_index, pos_item_real_index]
         
        # graph_wrappers
        graph_wrappers = []
        node_feature_info, edge_feature_info = [], []
        node_feature_info.append(('index', [None], np.dtype('int64')))
        node_feature_info.append(('term_ids', [None, None], np.dtype('int64')))
        for i in range(self.config.num_layers):
            graph_wrappers.append(
                pgl.graph_wrapper.GraphWrapper(
                    "layer_%s" % i, node_feat=node_feature_info, edge_feat=edge_feature_info))

        # encoder model
        encoder = Encoder.factory(self.config)
        outputs = encoder(graph_wrappers, [user_index, pos_item_index, neg_item_index])
        user_feat, pos_item_feat, neg_item_feat = outputs

        # loss 
        if self.config.neg_type == "batch_neg":
            neg_item_feat = pos_item_feat
        loss_func = Loss.factory(self.config)
        loss = loss_func(user_feat, pos_item_feat, neg_item_feat)

        # set datas, graph_wrappers, loss, outputs
        return datas, graph_wrappers, loss, outputs + [user_real_index, pos_item_real_index]


class NodeClassificationModel(BaseModel):

    def forward(self):
        # inputs
        node_index = L.data(
            "node_index", shape=[None], dtype="int64", append_batch_size=False)
        node_real_index = L.data(
            "node_real_index", shape=[None], dtype="int64", append_batch_size=False)
        label = L.data(
            "label", shape=[None], dtype="int64", append_batch_size=False)
        datas = [node_index, node_real_index, label]

        # graph_wrappers
        graph_wrappers = []
        node_feature_info = []
        node_feature_info.append(('index', [None], np.dtype('int64')))
        node_feature_info.append(('term_ids', [None, None], np.dtype('int64')))
        for i in range(self.config.num_layers):
            graph_wrappers.append(
                pgl.graph_wrapper.GraphWrapper(
                    "layer_%s" % i, node_feat=node_feature_info))

        # encoder model
        encoder = Encoder.factory(self.config)
        outputs = encoder(graph_wrappers, [node_index])
        feat = outputs[0]
        logits = L.fc(feat, self.config.num_label)

        # loss 
        label = L.reshape(label, [-1, 1])
        loss_func = Loss.factory(self.config)
        loss = loss_func(logits, label)

        return datas, graph_wrappers, loss, outputs + [node_real_index, logits]
