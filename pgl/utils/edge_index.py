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
"""This package implement EdgeIndex for Graph
"""
import os
import json
import paddle
import copy
import numpy as np
from pgl.utils import op
import pgl.graph_kernel as graph_kernel
from pgl.utils.helper import check_is_tensor


class EdgeIndex(object):
    """Indexing edges for fast graph queries
Sorted edges and represent edges in compressed style like csc_matrix or csr_matrix.

    Args:
        u: A list of node id to be compressed.
        v: A list of node id that are connected with u.
        num_nodes: The exactive number of nodes.
    """

    @classmethod
    def from_edges(cls, u, v, num_nodes):
        self = cls()
        self._is_tensor = check_is_tensor(u, v, num_nodes)
        if self._is_tensor:
            self._degree = paddle.zeros(shape=[num_nodes], dtype="int64")
            self._degree = paddle.scatter(
                x=self._degree,
                overwrite=False,
                index=u,
                updates=paddle.ones_like(
                    u, dtype="int64"))

            self._sorted_eid = paddle.argsort(u)
            self._sorted_u = paddle.gather(u, self._sorted_eid)
            self._sorted_v = paddle.gather(v, self._sorted_eid)
            self._indptr = op.get_index_from_counts(self._degree)
        else:
            self._degree, self._sorted_v, self._sorted_u, \
                self._sorted_eid, self._indptr = graph_kernel.build_index(u, v, num_nodes)
        return self

    @classmethod
    def from_index(cls, sorted_v, sorted_u, sorted_eid, degree, indptr):
        self = cls()
        self._degree = degree
        self._sorted_v = sorted_v
        self._sorted_u = sorted_u
        self._sorted_eid = sorted_eid
        self._indptr = indptr
        self._is_tensor = check_is_tensor(sorted_v, sorted_u, sorted_eid,
                                          degree, indptr)
        return self

    @classmethod
    def load(cls, path, mmap_mode="r"):
        """Load EdgeIndex from path and return a EdgeIndex in numpy. 

        Args:

            path: The directory path of the stored Graph.

            mmap_mode: Default :code:`mmap_mode="r"`. If not None, memory-map the graph.  
        """

        self = cls()
        self._degree = np.load(
            os.path.join(path, 'degree.npy'), mmap_mode=mmap_mode)
        self._sorted_u = np.load(
            os.path.join(path, 'sorted_u.npy'), mmap_mode=mmap_mode)
        self._sorted_v = np.load(
            os.path.join(path, 'sorted_v.npy'), mmap_mode=mmap_mode)
        self._sorted_eid = np.load(
            os.path.join(path, 'sorted_eid.npy'), mmap_mode=mmap_mode)
        self._indptr = np.load(
            os.path.join(path, 'indptr.npy'), mmap_mode=mmap_mode)
        self._is_tensor = False
        return self

    @property
    def degree(self):
        """Return the degree of nodes.
        """
        return self._degree

    def view_v(self, u=None):
        """Return the compressed v for given u.
        """
        if self._is_tensor:
            raise NotImplementedError("not implemented!")
        else:
            if u is None:
                return np.split(self._sorted_v, self._indptr[1:-1])
            else:
                u = np.array(u, dtype="int64")
                return graph_kernel.slice_by_index(
                    self._sorted_v, self._indptr, index=u)

    def view_eid(self, u=None):
        """Return the compressed edge id for given u.
        """
        if self._is_tensor:
            raise NotImplementedError("not implemented!")
        else:
            if u is None:
                return np.split(self._sorted_eid, self._indptr[1:-1])
            else:
                u = np.array(u, dtype="int64")
                return graph_kernel.slice_by_index(
                    self._sorted_eid, self._indptr, index=u)

    def triples(self):
        """Return the sorted (u, v, eid) tuples.
        """
        return self._sorted_u, self._sorted_v, self._sorted_eid

    def is_tensor(self):
        """Return whether the graph is paddle.Tensor or numpy.
        """
        return self._is_tensor

    def tensor(self, inplace=True):
        """Convert the EdgeIndex into paddle.Tensor format.

        In paddle.Tensor format, the graph edges and node features are in paddle.Tensor format.
        You can use send and recv in paddle.Tensor graph.
        
        Args:

            inplace: (Default True) Whether to convert the graph into tensor inplace. 
        
        """

        if self._is_tensor:
            # already tensor
            return self

        if inplace:
            self._sorted_u = paddle.to_tensor(self._sorted_u)
            self._sorted_v = paddle.to_tensor(self._sorted_v)
            self._sorted_eid = paddle.to_tensor(self._sorted_eid)
            self._degree = paddle.to_tensor(self._degree)
            self._indptr = paddle.to_tensor(self._indptr)
            self._is_tensor = True
            return self
        else:
            sorted_v = paddle.to_tensor(self._sorted_v)
            sorted_u = paddle.to_tensor(self._sorted_u)
            sorted_eid = paddle.to_tensor(self._sorted_eid)
            indptr = paddle.to_tensor(self._indptr)
            degree = paddle.to_tensor(self._degree)
            return EdgeIndex.from_index(
                sorted_v=sorted_v,
                sorted_u=sorted_u,
                sorted_eid=sorted_eid,
                indptr=indptr,
                degree=degree)

    def numpy(self, inplace=True):
        if not self._is_tensor:
            # already numpy
            return self

        if inplace:
            self._sorted_u = self._sorted_u.numpy()
            self._sorted_v = self._sorted_v.numpy()
            self._sorted_eid = self._sorted_eid.numpy()
            self._degree = paddle.to_tensor(self._degree)
            self._indptr = paddle.to_tensor(self._indptr)
            self._is_tensor = False
            return self
        else:
            sorted_v = self._sorted_v.numpy()
            sorted_u = self._sorted_u.numpy()
            sorted_eid = self._sorted_eid.numpy()
            degree = self._degree.numpy()
            indptr = self._indptr.numpy()
            return EdgeIndex.from_index(
                sorted_v=sorted_v,
                sorted_u=sorted_u,
                sorted_eid=sorted_eid,
                indptr=indptr,
                degree=degree)

    def dump(self, path):
        if self._is_tensor:
            edge_index = self.numpy(inplace=False)
            edge_index.dump(path)
        else:
            if not os.path.exists(path):
                os.makedirs(path)
            np.save(os.path.join(path, 'degree.npy'), self._degree)
            np.save(os.path.join(path, 'sorted_v.npy'), self._sorted_v)
            np.save(os.path.join(path, 'sorted_u.npy'), self._sorted_u)
            np.save(os.path.join(path, 'sorted_eid.npy'), self._sorted_eid)
            np.save(os.path.join(path, 'indptr.npy'), self._indptr)
