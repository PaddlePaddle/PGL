#-*- coding: utf-8 -*-
import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

import pgl
from pgl.nn import functional as GF

__all__ = ["LightGCNConv", ]


class LightGCNConv(nn.Layer):
    """
    
    Implementation of LightGCN
    
    This is an implementation of the paper LightGCN: Simplifying 
    and Powering Graph Convolution Network for Recommendation 
    (https://dl.acm.org/doi/10.1145/3397271.3401063).

    """

    def __init__(self):
        super(LightGCNConv, self).__init__()

    def forward(self, graph, feature):
        """
        Args:
 
            graph: `pgl.Graph` instance.

            feature: A tensor with shape (num_nodes, input_size)

        Return:

            A tensor with shape (num_nodes, output_size)

        """
        feature = graph.send_recv(feature, "sum")
        return feature
