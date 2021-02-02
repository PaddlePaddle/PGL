'''build label embedding model
'''
import math
import pgl
import paddle.fluid as F
import paddle.fluid.layers as L
from pgl.utils import paddle_helper
from module.transformer_gat_pgl import transformer_gat_pgl
from module.model_unimp_large import graph_transformer, linear, attn_appnp

class Arxiv_baseline_model():
    def __init__(self, gw, hidden_size, num_heads, dropout, num_layers):
        '''Arxiv_baseline_model
        '''
        self.gw=gw
        self.hidden_size=hidden_size
        self.num_heads= num_heads
        self.dropout= dropout
        self.num_layers=num_layers
        self.out_size=40
        self.embed_size=128  
        self.checkpoints=[]
        self.build_model()
    
    def embed_input(self, feature):
        
        lay_norm_attr = F.ParamAttr(initializer=F.initializer.ConstantInitializer(value=1))
        lay_norm_bias = F.ParamAttr(initializer=F.initializer.ConstantInitializer(value=0))
        feature = L.layer_norm(feature, name='layer_norm_feature_input', 
                                      param_attr=lay_norm_attr, 
                                      bias_attr=lay_norm_bias)
        
        return feature
        
   
    def build_model(self):
        
        feature_batch = self.embed_input(self.gw.node_feat['feat'])
        feature_batch = L.dropout(feature_batch, dropout_prob=self.dropout, 
                                dropout_implementation='upscale_in_train')
        for i in range(self.num_layers - 1):
            feature_batch = graph_transformer(str(i), self.gw, feature_batch, 
                                             hidden_size=self.hidden_size,
                                             num_heads=self.num_heads, 
                                             concat=True, skip_feat=True,
                                             layer_norm=True, relu=True, gate=True)
            if self.dropout > 0:
                feature_batch = L.dropout(feature_batch, dropout_prob=self.dropout, 
                                     dropout_implementation='upscale_in_train') 
            self.checkpoints.append(feature_batch)
        
        feature_batch = graph_transformer(str(self.num_layers - 1), self.gw, feature_batch, 
                                             hidden_size=self.out_size,
                                             num_heads=self.num_heads, 
                                             concat=False, skip_feat=True,
                                             layer_norm=False, relu=False, gate=True)
        self.checkpoints.append(feature_batch)
        self.out_feat = feature_batch
        
    def train_program(self,):
        label = F.data(name="label", shape=[None, 1], dtype="int64")
        train_idx = F.data(name='train_idx', shape=[None], dtype="int64")
        prediction = L.gather(self.out_feat, train_idx, overwrite=False)
        label = L.gather(label, train_idx, overwrite=False)
        cost = L.softmax_with_cross_entropy(logits=prediction, label=label)
        avg_cost = L.mean(cost)
        self.avg_cost = avg_cost
        
class Arxiv_label_embedding_model():
    def __init__(self, gw, hidden_size, num_heads, dropout, num_layers):
        '''Arxiv_label_embedding_model
        '''
        self.gw = gw
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.dropout = dropout
        self.num_layers = num_layers
        self.out_size = 40
        self.embed_size = 128 
        self.checkpoints = []
        self.build_model()
    
    def label_embed_input(self, feature):
        label = F.data(name="label", shape=[None, 1], dtype="int64")
        label_idx = F.data(name='label_idx', shape=[None], dtype="int64")
        label = L.reshape(label, shape=[-1])
        label = L.gather(label, label_idx, overwrite=False)
        
        lay_norm_attr = F.ParamAttr(initializer=F.initializer.ConstantInitializer(value=1))
        lay_norm_bias = F.ParamAttr(initializer=F.initializer.ConstantInitializer(value=0))
        feature = L.layer_norm(feature, name='layer_norm_feature_input1', 
                                      param_attr=lay_norm_attr, 
                                      bias_attr=lay_norm_bias)
        
        
        embed_attr = F.ParamAttr(initializer=F.initializer.NormalInitializer(loc=0.0, scale=1.0))
        embed = F.embedding(input=label, size=(self.out_size, self.embed_size), param_attr=embed_attr )
        lay_norm_attr = F.ParamAttr(initializer=F.initializer.ConstantInitializer(value=1))
        lay_norm_bias = F.ParamAttr(initializer=F.initializer.ConstantInitializer(value=0))
        embed = L.layer_norm(embed, name='layer_norm_feature_input2', 
                                      param_attr=lay_norm_attr, 
                                      bias_attr=lay_norm_bias)
        embed = L.relu(embed)
        
        feature_label = L.gather(feature, label_idx, overwrite=False)
        feature_label = feature_label + embed
        feature = L.scatter(feature, label_idx, feature_label, overwrite=True)
        
        return feature
        
    def build_model(self): 
        label_feature = self.label_embed_input(self.gw.node_feat['feat'])
        feature_batch = L.dropout(label_feature, dropout_prob=self.dropout, 
                                dropout_implementation='upscale_in_train')

        for i in range(self.num_layers - 1):
            feature_batch, _, cks = graph_transformer(str(i), self.gw, feature_batch, 
                                             hidden_size=self.hidden_size,
                                             num_heads=self.num_heads,
                                             attn_drop=True,
                                             concat=True, skip_feat=True,
                                             layer_norm=True, relu=True, gate=True)
            if self.dropout > 0:
                feature_batch = L.dropout(feature_batch, dropout_prob=self.dropout, 
                                     dropout_implementation='upscale_in_train') 
            self.checkpoints = self.checkpoints + cks
        
        feature_batch, attn, cks = graph_transformer(str(self.num_layers - 1), self.gw, feature_batch, 
                                             hidden_size=self.out_size,
                                             num_heads=self.num_heads+1, 
                                             concat=False, skip_feat=True,
                                             layer_norm=False, relu=False, gate=True)
        self.checkpoints.append(feature_batch)
        feature_batch = attn_appnp(self.gw, feature_batch, attn, alpha=0.2, k_hop=10)

        self.checkpoints.append(feature_batch)
        self.out_feat = feature_batch
        
    def train_program(self,):
        label = F.data(name="label", shape=[None, 1], dtype="int64")
        train_idx = F.data(name='train_idx', shape=[None], dtype="int64")
        prediction = L.gather(self.out_feat, train_idx, overwrite=False)
        label = L.gather(label, train_idx, overwrite=False)
        cost = L.softmax_with_cross_entropy(logits=prediction, label=label)
        avg_cost = L.mean(cost)
        self.avg_cost = avg_cost
    
