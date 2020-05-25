from paddle import fluid
from pgl.utils import paddle_helper
from pgl.layers import GaAN

class GaANModel(object):
    def __init__(self, num_class, num_layers, hidden_size_a=24, 
                 hidden_size_v=32, hidden_size_m=64, hidden_size_o=128,  
                 heads=8, act='relu', name="GaAN"):
        self.num_class = num_class
        self.num_layers = num_layers
        self.hidden_size_a = hidden_size_a
        self.hidden_size_v = hidden_size_v
        self.hidden_size_m = hidden_size_m
        self.hidden_size_o = hidden_size_o
        self.act = act
        self.name = name
        self.heads = heads    
    
    def forward(self, gw):
        feature = gw.node_feat['node_feat']
        for i in range(self.num_layers):
            feature = GaAN(gw, feature, self.hidden_size_a, self.hidden_size_v,
                                    self.hidden_size_m, self.hidden_size_o, self.heads, 
                                    self.name+'_'+str(i))
        
        pred = fluid.layers.fc(
            feature, self.num_class, act=None, name=self.name + "_pred_output")
        
        return pred
    


        
    
    
    