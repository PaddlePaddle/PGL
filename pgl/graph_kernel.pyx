# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved
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
from libcpp cimport bool

cdef extern from "stdint.h":
    ctypedef signed int int64_t

cdef extern from *:
    """
    #if defined(_WIN32) || defined(MS_WINDOWS) || defined(_MSC_VER)
        #include "third_party/metis/include/win.h"
        #define win32 1
        #define METIS_Recursive_(a,b,c,d,e,f,g,h,i,j,k,l,m) METIS_Recursive_win32(a,b,c,d,e,f,g,h,i,j,k,l,m)
        #define METIS_Kway_(a,b,c,d,e,f,g,h,i,j,k,l,m) METIS_Kway_win32(a,b,c,d,e,f,g,h,i,j,k,l,m)
        #define METIS_DefaultOptions_(m) METIS_DefaultOptions_win32(m)
    #else
        #include "third_party/metis/include/metis.h"
        #define win32 0
        #define METIS_Recursive_(a,b,c,d,e,f,g,h,i,j,k,l,m) METIS_PartGraphRecursive(a,b,c,d,e,f,g,h,i,j,k,l,m)
        #define METIS_Kway_(a,b,c,d,e,f,g,h,i,j,k,l,m) METIS_PartGraphKway(a,b,c,d,e,f,g,h,i,j,k,l,m)
        #define METIS_DefaultOptions_(m) METIS_SetDefaultOptions(m)
    #endif
    """
    bool win "win32"
    int METIS_Recursive "METIS_Recursive_"(int64_t *nvtxs, int64_t *ncon, int64_t *xadj,
                  int64_t *adjncy, int64_t *vwgt, int64_t *vsize, int64_t *adjwgt,
                  int64_t *nparts, float *tpwgts, float *ubvec, int64_t *options,
                  int64_t *edgecut, int64_t *part) nogil
    int METIS_Kway "METIS_Kway_"(int64_t *nvtxs, int64_t *ncon, int64_t *xadj,
                  int64_t *adjncy, int64_t *vwgt, int64_t *vsize, int64_t *adjwgt,
                  int64_t *nparts, float *tpwgts, float *ubvec, int64_t *options,
                  int64_t *edgecut, int64_t *part) nogil 
    int METIS_DefaultOptions "METIS_DefaultOptions_"(int64_t *options)


@cython.boundscheck(False)
@cython.wraparound(False)
def build_index(np.ndarray[np.int64_t, ndim=1] u,
        np.ndarray[np.int64_t, ndim=1] v,
        long long num_nodes):
    """Building Edge Index
    """
    cdef long long i
    cdef long long h=len(u)
    cdef long long n_size = num_nodes
    cdef np.ndarray[np.int64_t, ndim=1] degree = np.zeros([n_size], dtype=np.int64)
    cdef np.ndarray[np.int64_t, ndim=1] count = np.zeros([n_size], dtype=np.int64)
    cdef np.ndarray[np.int64_t, ndim=1] _tmp_v = np.zeros([h], dtype=np.int64)
    cdef np.ndarray[np.int64_t, ndim=1] _tmp_u = np.zeros([h], dtype=np.int64)
    cdef np.ndarray[np.int64_t, ndim=1] _tmp_eid = np.zeros([h], dtype=np.int64)
    cdef np.ndarray[np.int64_t, ndim=1] indptr = np.zeros([n_size + 1], dtype=np.int64)

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
    return degree, _tmp_v, _tmp_u, _tmp_eid, indptr

@cython.boundscheck(False)
@cython.wraparound(False)
def slice_by_index(np.ndarray[np.int64_t, ndim=1] u,
    np.ndarray[np.int64_t, ndim=1] indptr,
    np.ndarray[np.int64_t, ndim=1] index):
    cdef list output = []
    cdef long long i
    cdef long long h = len(index)
    cdef long long j
    for i in xrange(h):
        j = index[i] 
        output.append(u[indptr[j]:indptr[j+1]])
    return np.array(output, dtype=object)

@cython.boundscheck(False)
@cython.wraparound(False)
def map_edges(np.ndarray[np.int64_t, ndim=1] eid,
        np.ndarray[np.int64_t, ndim=2] edges,
        reindex):
    """Mapping edges by given dictionary
    """
    cdef unordered_map[long long, long long] m = reindex
    cdef long long i = 0
    cdef long long h = len(eid)
    cdef np.ndarray[np.int64_t, ndim=2] r_edges = np.zeros([h, 2], dtype=np.int64)
    cdef long long j
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
    cdef np.ndarray[np.int64_t, ndim=1] t_nodes = np.array(nodes, dtype=np.int64)
    cdef unordered_map[long long, long long] m = reindex
    cdef long long i = 0
    cdef long long h = len(nodes)
    cdef np.ndarray[np.int64_t, ndim=1] new_nodes = np.zeros([h], dtype=np.int64)
    cdef long long j
    with nogil:
        for i in xrange(h):
            j = t_nodes[i]
            new_nodes[i] = m[j]
    return new_nodes

@cython.boundscheck(False)
@cython.wraparound(False)
def node2vec_sample(np.ndarray[np.int64_t, ndim=1] succ,
        np.ndarray[np.int64_t, ndim=1] prev_succ, long long prev_node,
        float p, float q):
    """Fast implement of node2vec sampling
    """
    cdef long long i
    cdef succ_len = len(succ)
    cdef prev_succ_len = len(prev_succ)

    cdef vector[float] probs
    cdef float prob_sum = 0

    cdef unordered_set[long long] prev_succ_set
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

    cdef long long sample_succ = 0
    for i in xrange(succ_len):
        rand_num -= probs[i]
        if rand_num <= 0:
            sample_succ = succ[i]
            return sample_succ

@cython.boundscheck(False)
@cython.wraparound(False)
def node2vec_plus_sample(np.ndarray[np.int64_t, ndim=1] succ,
        np.ndarray[np.int64_t, ndim=1] prev_succ, long long prev_node,
        float p, float q):
    """Fast implement of node2vec sampling
    """
    cdef long long i
    cdef succ_len = len(succ)
    cdef prev_succ_len = len(prev_succ)

    cdef vector[float] probs
    cdef float prob_sum = 0

    cdef unordered_set[long long] prev_succ_set
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

    for i in xrange(succ_len):
        prev_succ_set.insert(succ[i])

    cdef int new_prev_succ_size = prev_succ_set.size()
    cdef np.ndarray[np.int64_t, ndim=1] new_prev_succ = np.zeros([new_prev_succ_size], dtype=np.int64)
    cdef int idx = 0
    for node in prev_succ_set:
        new_prev_succ[idx] = node
        idx += 1

    cdef float rand_num = float(rand())/RAND_MAX * prob_sum

    cdef long long sample_succ = 0
    for i in xrange(succ_len):
        rand_num -= probs[i]
        if rand_num <= 0:
            sample_succ = succ[i]
            return sample_succ, new_prev_succ


@cython.boundscheck(False)
@cython.wraparound(False)
def subset_choose_index(long long s_size,
                            np.ndarray[ndim=1, dtype=np.int64_t] nid,
                            np.ndarray[ndim=1, dtype=np.int64_t] rnd,
                            np.ndarray[ndim=1, dtype=np.int64_t] buff_nid,
                           long long offset):
    cdef long long n_size = len(nid)
    cdef long long i
    cdef long long j
    cdef unordered_map[long long, long long] m
    with nogil:
        for i in xrange(s_size):
            j = rnd[offset + i] % (n_size - i)
            buff_nid[offset + i] = nid[j] if m.find(j) == m.end() else nid[m[j]]
            m[j] = n_size - i - 1 if m.find(n_size - i - 1) == m.end() else m[n_size - i -1]


@cython.boundscheck(False)
@cython.wraparound(False)
def subset_choose_index_eid(long long s_size,
                            np.ndarray[ndim=1, dtype=np.int64_t] nid,
                            np.ndarray[ndim=1, dtype=np.int64_t] eid,
                            np.ndarray[ndim=1, dtype=np.int64_t] rnd,
                            np.ndarray[ndim=1, dtype=np.int64_t] buff_nid,
                            np.ndarray[ndim=1, dtype=np.int64_t] buff_eid,
                           long long offset):
    cdef long long n_size = len(nid)
    cdef long long i
    cdef long long j
    cdef unordered_map[long long, long long] m
    with nogil:
        for i in xrange(s_size):
            j = rnd[offset + i] % (n_size - i)
            buff_nid[offset + i] = nid[j] if m.find(j) == m.end() else nid[m[j]]
            buff_eid[offset + i] = eid[j] if m.find(j) == m.end() else eid[m[j]]
            m[j] = n_size - i - 1 if m.find(n_size - i - 1) == m.end() else m[n_size - i -1]


@cython.boundscheck(False)
@cython.wraparound(False)
def sample_subset(list nids, long long maxdegree, shuffle=False):
    cdef np.ndarray[ndim=1, dtype=np.int64_t] buff_index
    cdef long long buff_size, sample_size
    cdef long long total_buff_size = 0
    cdef long long inc = 0
    cdef list output = []
    for inc in xrange(len(nids)):
        buff_size = len(nids[inc])
        if buff_size > maxdegree:
            total_buff_size += maxdegree
        elif shuffle:
            total_buff_size += buff_size
    cdef np.ndarray[ndim=1, dtype=np.int64_t] buff_nid = np.zeros([total_buff_size], dtype=np.int64)
    cdef np.ndarray[np.int64_t, ndim=1] rnd = np.random.randint(0,  np.iinfo(np.int64).max,
                                                              dtype=np.int64, size=total_buff_size)

    cdef long long offset = 0
    for inc in xrange(len(nids)):
        buff_size = len(nids[inc])
        if not shuffle and buff_size <= maxdegree:
            output.append(nids[inc])
        else:
            sample_size = buff_size if buff_size <= maxdegree else maxdegree
            if isinstance(nids[inc], list):
                tmp = np.array(nids[inc], dtype=np.int64)
            else:
                tmp = nids[inc]
            subset_choose_index(sample_size, tmp, rnd, buff_nid, offset)
            output.append(buff_nid[offset:offset+sample_size])
            offset += sample_size
    return output

@cython.boundscheck(False)
@cython.wraparound(False)
def sample_subset_with_eid(list nids, list eids, long long maxdegree, shuffle=False):
    cdef np.ndarray[ndim=1, dtype=np.int64_t] buff_index
    cdef long long buff_size, sample_size
    cdef long long total_buff_size = 0
    cdef long long inc = 0
    cdef list output = []
    cdef list output_eid = []
    for inc in xrange(len(nids)):
        buff_size = len(nids[inc])
        if buff_size > maxdegree:
            total_buff_size += maxdegree
        elif shuffle:
            total_buff_size += buff_size
    cdef np.ndarray[ndim=1, dtype=np.int64_t] buff_nid = np.zeros([total_buff_size], dtype=np.int64)
    cdef np.ndarray[ndim=1, dtype=np.int64_t] buff_eid = np.zeros([total_buff_size], dtype=np.int64)
    cdef np.ndarray[np.int64_t, ndim=1] rnd = np.random.randint(0,  np.iinfo(np.int64).max,
                                                              dtype=np.int64, size=total_buff_size)

    cdef long long offset = 0
    for inc in xrange(len(nids)):
        buff_size = len(nids[inc])
        if not shuffle and buff_size <= maxdegree:
            output.append(nids[inc])
            output_eid.append(eids[inc])
        else:
            sample_size = buff_size if buff_size <= maxdegree else maxdegree
            if isinstance(nids[inc], list):
                tmp = np.array(nids[inc], dtype=np.int64)
                tmp_eids = np.array(eids[inc], dtype=np.int64)
            else:
                tmp = nids[inc]
                tmp_eids = eids[inc]

            subset_choose_index_eid(sample_size, tmp, tmp_eids, rnd, buff_nid, buff_eid, offset)
            output.append(buff_nid[offset:offset+sample_size])
            output_eid.append(buff_eid[offset:offset+sample_size])
            offset += sample_size
    return output, output_eid

@cython.boundscheck(False)
@cython.wraparound(False)
def skip_gram_gen_pair(vector[long long] walk, long win_size=5):
    cdef vector[long long] src
    cdef vector[long long] dst
    cdef long long l = len(walk)
    cdef long long real_win_size, left, right, i
    cdef np.ndarray[np.int64_t, ndim=1] rnd = np.random.randint(1,  win_size+1,
                                    dtype=np.int64, size=l)
    with nogil:
        for i in xrange(l):
            real_win_size = rnd[i]
            left = i - real_win_size
            if left < 0:
                left = 0
            right = i + real_win_size
            if right >= l:
                right = l - 1
            for j in xrange(left, right+1):
                if walk[i] == walk[j]:
                    continue
                src.push_back(walk[i])
                dst.push_back(walk[j])
    return src, dst

@cython.boundscheck(False)
@cython.wraparound(False)
def alias_sample_build_table(np.ndarray[np.float64_t, ndim=1] probs):
    cdef long long l = len(probs)
    cdef np.ndarray[np.float64_t, ndim=1] alias = probs * l
    cdef np.ndarray[np.int64_t, ndim=1] events = np.zeros(l, dtype=np.int64)

    cdef vector[long long] larger_num, smaller_num
    cdef long long i, s_i, l_i
    with nogil:
        for i in xrange(l):
            if alias[i] > 1:
                larger_num.push_back(i)
            elif alias[i] < 1:
                smaller_num.push_back(i)

        while smaller_num.size() > 0 and larger_num.size() > 0:
            s_i = smaller_num.back()
            l_i = larger_num.back()
            smaller_num.pop_back()
            events[s_i] = l_i
            alias[l_i] -= (1 - alias[s_i])
            if alias[l_i] <= 1:
                larger_num.pop_back()
            if alias[l_i] < 1:
                smaller_num.push_back(l_i)
    return alias, events

@cython.boundscheck(False)
@cython.wraparound(False)
def extract_edges_from_nodes(
    np.ndarray[np.int64_t, ndim=1] adj_indptr,
    np.ndarray[np.int64_t, ndim=1] sorted_v,
    np.ndarray[np.int64_t, ndim=1] sorted_eid,
    vector[long long] sampled_nodes,
):
    """
    Extract all eids of given sampled_nodes for the origin graph.
    ret_edge_index: edge ids between sampled_nodes.

    Refers: https://github.com/GraphSAINT/GraphSAINT
    """
    cdef long long i, v, j
    cdef long long num_v_orig, num_v_sub
    cdef long long start_neigh, end_neigh
    cdef vector[int] _arr_bit
    cdef vector[long long] ret_edge_index
    num_v_orig = adj_indptr.size-1
    _arr_bit = vector[int](num_v_orig,-1)
    num_v_sub = sampled_nodes.size()
    i = 0
    with nogil:
        while i < num_v_sub:
            _arr_bit[sampled_nodes[i]] = i
            i = i + 1
        i = 0
        while i < num_v_sub:
            v = sampled_nodes[i]
            start_neigh = adj_indptr[v]
            end_neigh = adj_indptr[v+1]
            j = start_neigh
            while j < end_neigh:
                if _arr_bit[sorted_v[j]] > -1:
                    ret_edge_index.push_back(sorted_eid[j])
                j = j + 1
            i = i + 1
    return ret_edge_index
   
@cython.boundscheck(False)
@cython.wraparound(False)
def metis_partition(
    int64_t num_nodes,
    np.ndarray[np.int64_t, ndim=1] adj_indptr,
    np.ndarray[np.int64_t, ndim=1] sorted_v,
    int64_t nparts,
    np.ndarray[np.int64_t, ndim=1] node_weights=None,
    np.ndarray[np.int64_t, ndim=1] edge_weights=None,
    bool recursive=True,
):
    cdef int64_t edgecut = -1
    cdef int64_t ncon = 1

    cdef np.ndarray part = np.zeros((num_nodes, ), dtype="int64")

    cdef int64_t * node_weight_ptr = NULL

    if node_weights is not None:
        node_weight_ptr = <int64_t *> node_weights.data

    cdef int64_t * edge_weight_ptr = NULL
    if edge_weights is not None:
        edge_weight_ptr = <int64_t *> edge_weights.data


    if win == 0:
        with nogil:
            if recursive:
                METIS_Recursive(nvtxs=&num_nodes, ncon=&ncon, xadj=<int64_t *> adj_indptr.data,
                             adjncy=<int64_t *> sorted_v.data, vwgt=node_weight_ptr, vsize=NULL, adjwgt=edge_weight_ptr,
                             nparts=&nparts, tpwgts=NULL, ubvec=NULL, options=NULL,
                             edgecut=&edgecut, part=<int64_t *> part.data)
            else:
                METIS_Kway(nvtxs=&num_nodes, ncon=&ncon, xadj=<int64_t *> adj_indptr.data,
                             adjncy=<int64_t *> sorted_v.data, vwgt=node_weight_ptr, vsize=NULL, adjwgt=edge_weight_ptr,
                             nparts=&nparts, tpwgts=NULL, ubvec=NULL, options=NULL,
                             edgecut=&edgecut, part=<int64_t *> part.data)
    return part
    

