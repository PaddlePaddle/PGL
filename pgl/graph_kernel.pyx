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
"""
    Fast implementation for graph construction and sampling.
"""
import numpy as np
cimport numpy as np
cimport cython
from libcpp.map cimport map
from libcpp.set cimport set
from libcpp.unordered_set cimport unordered_set
from libcpp.unordered_map cimport unordered_map
from libcpp.vector cimport vector
from libc.stdlib cimport rand, RAND_MAX

@cython.boundscheck(False)
@cython.wraparound(False)
def build_index(np.ndarray[np.int32_t, ndim=1] u,
        np.ndarray[np.int32_t, ndim=1] v,
        int num_nodes):
    """Building Edge Index
    """
    cdef int i
    cdef int h=len(u)
    cdef int n_size = num_nodes
    cdef np.ndarray[np.int32_t, ndim=1] degree = np.zeros([n_size], dtype=np.int32)
    cdef np.ndarray[np.int32_t, ndim=1] count = np.zeros([n_size], dtype=np.int32)
    cdef np.ndarray[np.int32_t, ndim=1] _tmp_v = np.zeros([h], dtype=np.int32)
    cdef np.ndarray[np.int32_t, ndim=1] _tmp_u = np.zeros([h], dtype=np.int32)
    cdef np.ndarray[np.int32_t, ndim=1] _tmp_eid = np.zeros([h], dtype=np.int32)
    cdef np.ndarray[np.int32_t, ndim=1] indptr = np.zeros([n_size + 1], dtype=np.int32)

    with nogil:
        for i in xrange(h):
            degree[u[i]] += 1

        for i in xrange(n_size):
            indptr[i + 1] = indptr[i] + degree[i]

        for i in xrange(h):
            _tmp_v[indptr[u[i]] + count[u[i]]] = v[i]
            _tmp_eid[indptr[u[i]] + count[u[i]]] = i
            _tmp_u[indptr[u[i]] + count[u[i]]] = u[i]
            count[u[i]] += 1

    cdef list output_eid = []
    cdef list output_v = []
    for i in xrange(n_size):
        output_eid.append(_tmp_eid[indptr[i]:indptr[i+1]])
        output_v.append(_tmp_v[indptr[i]:indptr[i+1]])
    return np.array(output_v), np.array(output_eid), degree, _tmp_u, _tmp_v, _tmp_eid


@cython.boundscheck(False)
@cython.wraparound(False)
def map_edges(np.ndarray[np.int32_t, ndim=1] eid,
        np.ndarray[np.int32_t, ndim=2] edges,
        reindex):
    """Mapping edges by given dictionary
    """
    cdef unordered_map[int, int] m = reindex
    cdef int i = 0
    cdef int h = len(eid)
    cdef np.ndarray[np.int32_t, ndim=2] r_edges = np.zeros([h, 2], dtype=np.int32)
    cdef int j
    with nogil:
        for i in xrange(h):
            j = eid[i]
            r_edges[i, 0] = m[edges[j, 0]]
            r_edges[i, 1] = m[edges[j, 1]]
    return r_edges

@cython.boundscheck(False)
@cython.wraparound(False)
def map_nodes(nodes, reindex):
    """Mapping nodes by given dictionary
    """
    cdef unordered_map[int, int] m = reindex
    cdef int i = 0
    cdef int h = len(nodes)
    cdef np.ndarray[np.int32_t, ndim=1] new_nodes = np.zeros([h], dtype=np.int32)
    cdef int j
    for i in xrange(h):
        j = nodes[i]
        new_nodes[i] = m[j]
    return new_nodes

@cython.boundscheck(False)
@cython.wraparound(False)
def node2vec_sample(np.ndarray[np.int32_t, ndim=1] succ,
        np.ndarray[np.int32_t, ndim=1] prev_succ, int prev_node, 
        float p, float q):
    """Fast implement of node2vec sampling
    """
    cdef int i
    cdef succ_len = len(succ)
    cdef prev_succ_len = len(prev_succ)

    cdef vector[float] probs
    cdef float prob_sum = 0

    cdef unordered_set[int] prev_succ_set
    for i in xrange(prev_succ_len):
        prev_succ_set.insert(prev_succ[i])

    cdef float prob
    for i in xrange(succ_len):
        if succ[i] == prev_node:
            prob = 1. / p
        elif prev_succ_set.find(succ[i]) != prev_succ_set.end():
            prob = 1.
        else:
            prob = 1. / q
        probs.push_back(prob)
        prob_sum += prob

    cdef float rand_num = float(rand())/RAND_MAX * prob_sum

    cdef int sample_succ = 0
    for i in xrange(succ_len):
        rand_num -= probs[i]
        if rand_num <= 0:
            sample_succ = succ[i]
            return sample_succ
