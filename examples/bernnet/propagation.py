import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import pgl
from pgl.nn import functional as GF
import math
from scipy.special import comb

class bern_prop(nn.Layer):
    def __init__(
        self,
        k_hop=10,
    ):
        super(bern_prop, self).__init__()
        self.K = k_hop
        self.temp = self.create_parameter(
            shape=[self.K+1],
            dtype="float32",
            default_initializer=nn.initializer.Constant(value=1.0),
        )

    def forward(self, graph, feature, norm=None):
        TEMP=F.relu(self.temp)
        norm = GF.degree_norm(graph)
        tmp=[]
        tmp.append(feature)
        for i in range(self.K):
            h0 = feature
            feature = feature*norm
            feature = graph.send_recv(feature)
            feature = feature*norm
            feature = h0+feature
            tmp.append(feature)
        out=(comb(self.K,0)/(2**self.K))*TEMP[0]*tmp[self.K]
        for i in range(self.K):
            feature=tmp[self.K-i-1]
            h0 = feature
            feature = feature*norm
            feature = graph.send_recv(feature)
            feature = feature*norm
            feature = h0-feature
            for j in range(i):
                h0 = feature
                feature = feature*norm
                feature = graph.send_recv(feature)
                feature = feature*norm
                feature = h0-feature
            out=out+(comb(self.K,i+1)/(2**self.K))*TEMP[i+1]*feature
        return out