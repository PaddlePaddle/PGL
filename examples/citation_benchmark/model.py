import pgl
import paddle.fluid.layers as L
import pgl.layers.conv as conv

def get_norm(indegree):
    norm = L.pow(L.cast(indegree, dtype="float32") + 1e-6, factor=-0.5) 
    norm = norm * L.cast(indegree > 0, dtype="float32")
    return norm
    

class GCN(object):
    """Implement of GCN
    """
    def __init__(self, config, num_class):
        self.num_class = num_class
        self.num_layers = config.get("num_layers", 1)
        self.hidden_size = config.get("hidden_size", 64)
        self.dropout = config.get("dropout", 0.5)
        self.edge_dropout = config.get("edge_dropout", 0.0)

    def forward(self, graph_wrapper, feature, phase):
        
        for i in range(self.num_layers):

            if phase == "train":
                ngw = pgl.sample.edge_drop(graph_wrapper, self.edge_dropout) 
                norm = get_norm(ngw.indegree())
            else:
                ngw = graph_wrapper
                norm = graph_wrapper.node_feat["norm"]

            feature = L.dropout(
                    feature,
                    self.dropout,
                    dropout_implementation='upscale_in_train')

            feature = pgl.layers.gcn(ngw,
                feature,
                self.hidden_size,
                activation="relu",
                norm=norm,
                name="layer_%s" % i)

        feature = L.dropout(
                    feature,
                    self.dropout,
                    dropout_implementation='upscale_in_train')

        if phase == "train": 
            ngw = pgl.sample.edge_drop(graph_wrapper, self.edge_dropout) 
            norm = get_norm(ngw.indegree())
        else:
            ngw = graph_wrapper
            norm = graph_wrapper.node_feat["norm"]

        feature = conv.gcn(ngw,
                     feature,
                     self.num_class,
                     activation=None,
                     norm=norm,
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
        self.edge_dropout = config.get("edge_dropout", 0.0)

    def forward(self, graph_wrapper, feature, phase):
        if phase == "train": 
            edge_dropout = 0
        else:
            edge_dropout = self.edge_dropout

        for i in range(self.num_layers):
            ngw = pgl.sample.edge_drop(graph_wrapper, edge_dropout) 
                
            feature = conv.gat(ngw,
                                feature,
                                self.hidden_size,
                                activation="elu",
                                name="gat_layer_%s" % i,
                                num_heads=self.num_heads,
                                feat_drop=self.feat_dropout,
                                attn_drop=self.attn_dropout)

        ngw = pgl.sample.edge_drop(graph_wrapper, edge_dropout) 
        feature = conv.gat(ngw,
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
        self.edge_dropout = config.get("edge_dropout", 0.0)

    def forward(self, graph_wrapper, feature, phase):
        if phase == "train": 
            edge_dropout = 0
        else:
            edge_dropout = self.edge_dropout

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
            edge_dropout=edge_dropout,
            alpha=self.alpha,
            k_hop=self.k_hop)
        return feature

class SGC(object):
    """Implement of SGC"""
    def __init__(self, config, num_class):
        self.num_class = num_class
        self.num_layers = config.get("num_layers", 1)

    def forward(self, graph_wrapper, feature, phase):
        feature = conv.appnp(graph_wrapper,
            feature=feature,
            norm=graph_wrapper.node_feat["norm"],
            alpha=0,
            k_hop=self.num_layers)
        feature.stop_gradient=True
        feature = L.fc(feature, self.num_class, act=None, name="output")
        return feature

