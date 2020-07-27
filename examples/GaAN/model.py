# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved
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
from paddle import fluid
from pgl.utils import paddle_helper

# from pgl.layers import gaan
from conv import gaan

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
            feature = gaan(gw, feature, self.hidden_size_a, self.hidden_size_v,
                           self.hidden_size_m, self.hidden_size_o, self.heads, 
                           self.name+'_'+str(i))
        
        pred = fluid.layers.fc(
            feature, self.num_class, act=None, name=self.name + "_pred_output")
        
        return pred
    


        
    
    
    