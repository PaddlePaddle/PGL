import pgl
import paddle.fluid.layers as L
import pgl.layers.conv as conv

class GCN(object):
    """Implement of GCN
    """
    def __init__(self, config, num_class):
        self.num_class = num_class
        self.num_layers = config.get("num_layers", 1)
        self.hidden_size = config.get("hidden_size", 64)
        self.dropout = config.get("dropout", 0.5)

    def forward(self, graph_wrapper, feature):
        for i in range(self.num_layers):
            feature = pgl.layers.gcn(graph_wrapper,
                feature,
                self.hidden_size,
                activation="relu",
                norm=graph_wrapper.node_feat["norm"],
                name="layer_%s" % i)

            feature = L.dropout(
                    feature,
                    self.dropout,
                    dropout_implementation='upscale_in_train')

        feature = conv.gcn(graph_wrapper,
                     feature,
                     self.num_class,
                     activation=None,
                     norm=graph_wrapper.node_feat["norm"],
                     name="output")

        return feature

class GAT(object):
    """Implement of GAT"""
    def __init__(self, config, num_class):
        self.num_class = num_class 
        self.num_layers = config.get("num_layers", 1)
        self.num_heads = config.get("num_heads", 8)
        self.hidden_size = config.get("hidden_size", 8)
        self.feat_dropout = config.get("feat_drop", 0.6)
        self.attn_dropout = config.get("attn_drop", 0.6)

    def forward(self, graph_wrapper, feature):
        for i in range(self.num_layers):
            feature = conv.gat(graph_wrapper,
                                feature,
                                self.hidden_size,
                                activation="elu",
                                name="gat_layer_%s" % i,
                                num_heads=self.num_heads,
                                feat_drop=self.feat_dropout,
                                attn_drop=self.attn_dropout)

        feature = conv.gat(graph_wrapper,
                     feature,
                     self.num_class,
                     num_heads=1,
                     activation=None,
                     feat_drop=self.feat_dropout,
                     attn_drop=self.attn_dropout,
                     name="output")
        return feature

   
class APPNP(object):
    """Implement of APPNP"""
    def __init__(self, config, num_class):
        self.num_class = num_class
        self.num_layers = config.get("num_layers", 1)
        self.hidden_size = config.get("hidden_size", 64)
        self.dropout = config.get("dropout", 0.5)
        self.alpha = config.get("alpha", 0.1)
        self.k_hop = config.get("k_hop", 10)

    def forward(self, graph_wrapper, feature):
        for i in range(self.num_layers):
            feature = L.dropout(
                feature,
                self.dropout,
                dropout_implementation='upscale_in_train')
            feature = L.fc(feature, self.hidden_size, act="relu", name="lin%s" % i)

        feature = L.dropout(
            feature,
            self.dropout,
            dropout_implementation='upscale_in_train')

        feature = L.fc(feature, self.num_class, act=None, name="output")

        feature = conv.appnp(graph_wrapper,
            feature=feature,
            norm=graph_wrapper.node_feat["norm"],
            alpha=self.alpha,
            k_hop=self.k_hop)
        return feature

