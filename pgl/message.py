# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
import pgl.math as math


class Message(object):
    """This implement Message for graph.recv.

    Args:

        msg: A dictionary provided by send function.

        segment_ids: The id that the message belongs to.
   
    """

    def __init__(self, msg, segment_ids):
        self._segment_ids = segment_ids
        self._msg = msg

    def reduce(self, msg, pool_type="sum"):
        """This method reduce message by given `pool_type`.

        Now, this method only supports default reduce function, 
        with ('sum', 'mean', 'max', 'min').

        Args:

            feature (paddle.Tensor): feature with first dim as num_edges.

            pool_type (str): 'sum', 'mean', 'max', 'min' built-in receive function.

        Returns:
 
            Returns a paddle.Tensor with the first dim the same as the largest segment_id.
        """

        outputs = math.segment_pool(
            msg, self._segment_ids, pool_type=pool_type)
        return outputs

    def reduce_sum(self, msg):
        """This method reduce message by sum. 

        Args:

            feature (paddle.Tensor): feature with first dim as num_edges.

        Returns:
 
            Returns a paddle.Tensor with the first dim the same as the largest segment_id.
        """
        return math.segment_sum(msg, self._segment_ids)

    def reduce_mean(self, msg):
        """This method reduce message by mean. 

        Args:

            feature (paddle.Tensor): feature with first dim as num_edges.

        Returns:
 
            Returns a paddle.Tensor with the first dim the same as the largest segment_id.
        """
        return math.segment_mean(msg, self._segment_ids)

    def reduce_max(self, msg):
        """This method reduce message by max. 

        Args:

            feature (paddle.Tensor): feature with first dim as num_edges.

        Returns:
 
            Returns a paddle.Tensor with the first dim the same as the largest segment_id.
        """
        return math.segment_max(msg, self._segment_ids)

    def reduce_min(self, msg):
        """This method reduce message by min. 

        Args:

            feature (paddle.Tensor): feature with first dim as num_edges.

        Returns:
 
            Returns a paddle.Tensor with the first dim the same as the largest segment_id.
        """
        return math.segment_min(msg, self._segment_ids)

    def edge_expand(self, msg):
        """This is the inverse method for reduce.

        Args:

            feature (paddle.Tensor): A reduced message. 

        Returns:
 
            Returns a paddle.Tensor with the first dim the same as the num_edges.

        Examples:

            .. code-block:: python

                import numpy as np
                import pgl
                import paddle

                num_nodes = 5
                edges = [ (0, 1), (1, 2), (3, 4)]
                feature = np.random.randn(5, 100)
                edge_feature = np.random.randn(3, 100)
                graph = pgl.Graph(num_nodes=num_nodes,
                        edges=edges,
                        node_feat={
                            "feature": feature
                        },
                        edge_feat={
                            "edge_feature": edge_feature
                        })
                graph.tensor() 

                def send_func(src_feat, dst_feat, edge_feat):
                    return { "out": src_feat["feature"] }

                message = graph.send(send_func, src_feat={"feature": graph.node_feat["feature"]})

                def recv_func(msg):
                    value = msg["out"]
                    max_value = msg.reduce_max(value)
                    # We want to subscribe the max_value correspond to the destination node.
                    max_value = msg.edge_expand(max_value)
                    value = value - max_value
                    return msg.reduce_sum(value)
                   
                out = graph.recv(recv_func, message)

        """

        return paddle.gather(msg, self._segment_ids, axis=0)

    def reduce_softmax(self, msg):
        """This method reduce message by softmax. 

        Args:

            feature (paddle.Tensor): feature with first dim as num_edges.

        Returns:
 
            Returns a paddle.Tensor with the first dim the same as the largest segment_id.
        """
        return math.segment_softmax(msg, self._segment_ids)

    def __getitem__(self, key):
        return self._msg[key]
