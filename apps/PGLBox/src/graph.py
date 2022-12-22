# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
"""Distributed GPU Graph config
"""
import sys
import time

from paddle.fluid.core import GraphGpuWrapper
from pgl.utils.logger import log

import helper
import util
from place import get_cuda_places


class DistGraph(object):
    """ Initialize the Distributed Graph Server

    Args:
        root_dir: the graph data dir 
    
        node_types: the node type configs
 
        edge_types: the edge type configs.

        symmetry: whether the edges are symmetry

        slots: the node feature slot 

        token_slot: for erniesage token slot input

        slot_num_for_pull_feature: total slot feature number we should pull

        num_parts: the sharded parts of graph data

        metapath_split_opt: whether use metapath split optimization
    """

    def __init__(self,
                 root_dir,
                 node_types,
                 edge_types,
                 symmetry,
                 slots,
                 token_slot,
                 slot_num_for_pull_feature,
                 num_parts,
                 metapath_split_opt=False):
        self.root_dir = root_dir
        self.node_types = node_types
        self.edge_types = edge_types
        self.symmetry = symmetry
        self.slots = slots
        self.token_slot = token_slot
        self.slot_num_for_pull_feature = slot_num_for_pull_feature
        self.num_parts = num_parts
        self.metapath_split_opt = metapath_split_opt
        self.reverse = 1 if self.symmetry else 0

        self.etype2files = helper.parse_files(self.edge_types)
        self.etype_list = util.get_all_edge_type(self.etype2files, self.symmetry)

        self.ntype2files = helper.parse_files(self.node_types)
        self.ntype_list = list(self.ntype2files.keys())
        log.info("total etype: %s" % repr(self.etype_list))
        log.info("total ntype: %s" % repr(self.ntype_list))

        self.graph = GraphGpuWrapper()

        self._setup_graph()

    def _setup_graph(self):
        self.graph.set_device(get_cuda_places())
        self.graph.set_up_types(self.etype_list, self.ntype_list)

        for ntype in self.ntype_list:
            for slot_id in self.slots:
                log.info("add_table_feat_conf of slot id %s" % slot_id)
                self.graph.add_table_feat_conf(ntype,
                                               str(slot_id), "feasign", 1)

            if self.token_slot:
                log.info("add_table_feat_conf of token slot id %s" %
                         self.token_slot)
                self.graph.add_table_feat_conf(ntype,
                                               str(self.token_slot), "feasign",
                                               1)
        self.graph.set_slot_feature_separator(":")
        self.graph.set_feature_separator(",")
        self.graph.init_service()

    def load_edge(self):
        """Pull whole graph edges from disk into cpu memory, then load into gpu.
           After that, release memory on cpu.
        """
        cpuload_begin = time.time()
        load_begin_time = time.time()
        sys.stderr.write("begin load edge\n")
        self.graph.load_edge_file(self.edge_types, self.root_dir,
                                  self.num_parts, self.reverse)
        sys.stderr.write("end load edge\n")
        load_end_time = time.time()
        log.info("STAGE [CPU LOAD EDGE] finished, time cost: %f sec" %
                 (load_end_time - load_begin_time))

        if not self.metapath_split_opt:
            load_begin_time = time.time()
            for i in range(len(self.etype_list)):
                self.graph.upload_batch(0, i,
                                        len(get_cuda_places()),
                                        self.etype_list[i])
                sys.stderr.write("end upload edge, type[" + self.etype_list[i]
                                 + "]\n")
            load_end_time = time.time()
            log.info("STAGE [GPU LOAD EDGE] finished, time cost: %f sec" %
                     (load_end_time - load_begin_time))

        release_begin_time = time.time()
        self.graph.release_graph_edge()
        release_end_time = time.time()
        log.info("STAGE [CPU RELEASE EDGE] finished, time cost: %f sec" %
                 (release_end_time - release_begin_time))

    def load_node(self):
        """Pull whole graph nodes from disk into cpu memory, then load into gpu.
           After that, release memory on cpu.
        """
        load_begin_time = time.time()
        sys.stderr.write("begin load node\n")
        ret = self.graph.load_node_file(self.node_types, self.root_dir,
                                        self.num_parts)

        if ret is not 0:
            log.info("Fail to load node, ntype2files[%s] path[%s] num_part[%d]" \
                     % (self.node_types, self.root_dir, self.num_parts))
            return -1

        sys.stderr.write("end load node\n")
        load_end_time = time.time()
        log.info("STAGE [CPU LOAD NODE] finished, time cost: %f sec" %
                 (load_end_time - load_begin_time))

        load_begin_time = time.time()
        if self.slot_num_for_pull_feature > 0:
            self.graph.upload_batch(1,
                                    len(get_cuda_places()),
                                    self.slot_num_for_pull_feature)
        load_end_time = time.time()
        log.info("STAGE [GPU LOAD NODE] finished, time cost: %f sec" %
                 (load_end_time - load_begin_time))

        release_begin_time = time.time()
        self.graph.release_graph_node()
        release_end_time = time.time()
        log.info("STAGE [CPU RELEASE NODE] finished, time cost: %f sec" %
                 (release_end_time - release_begin_time))

        return 0

    def load_metapath_edges(self, metapath_dict, metapath):
        """Pull specific metapath's edges.
        """
        first_node = metapath.split('2')[0]
        all_metapath_class_len = len(metapath_dict[first_node])
        cur_metapath_index = metapath_dict[first_node].index(metapath)
        log.info('metapath: %s, first node:%s, all_len:%s, index:%s' %
                 (metapath, first_node, all_metapath_class_len, cur_metapath_index))

        sub_edge_types = metapath.split('-')
        edge_len = len(sub_edge_types)
        sub_etype2files = util.get_sub_path(self.edge_types, sub_edge_types, True)

        metapath_cpuload_begin = time.time()
        self.graph.load_edge_file(sub_etype2files, self.root_dir,
                                  self.num_parts, self.reverse)
        log.info("sub_etype2files: %s" % sub_etype2files)
        metapath_cpuload_end = time.time()
        log.info("metapath: %s, cpuload time: %s" % (metapath,
                 metapath_cpuload_end - metapath_cpuload_begin))

        metapath_gpuload_begin = time.time()
        for j in range(0, len(sub_edge_types)):
            log.info("begin upload edge type: %s" % sub_edge_types[j])
            self.graph.upload_batch(0, j, len(get_cuda_places()), sub_edge_types[j])
            log.info("end upload edge type: %s" % sub_edge_types[j])
        metapath_gpuload_end = time.time()
        log.info("metapath: %s, gpuload time:%s" % (metapath,
                 metapath_gpuload_end - metapath_gpuload_begin))

        self.graph.init_metapath(metapath, cur_metapath_index, all_metapath_class_len)

    def get_sorted_metapath_and_dict(self, metapaths):
        """
        """
        first_node_type = util.get_first_node_type(metapaths)
        node_type_size = self.graph.get_node_type_size(first_node_type)
        edge_type_size = self.graph.get_edge_type_size()
        sorted_metapaths = util.change_metapath_index(metapaths, node_type_size, edge_type_size)
        log.info("after change metapaths: %s" % sorted_metapaths)

        metapath_dict = {}
        for i in range(len(sorted_metapaths)):
            first_node = sorted_metapaths[i].split('2')[0]
            if first_node in metapath_dict:
                metapath_dict[first_node].append(sorted_metapaths[i])
            else:
                metapath_dict[first_node] = []
                metapath_dict[first_node].append(sorted_metapaths[i])
        return sorted_metapaths, metapath_dict

    def load_graph_into_cpu(self):
        """Pull whole graph from disk into memory
        """
        cpuload_begin = time.time()
        self.graph.load_node_and_edge(self.edge_types, self.node_types,
                                      self.root_dir, self.num_parts, self.reverse)
        cpuload_end = time.time()
        log.info("STAGE [CPU LOAD] finished, time cost: %f sec",
                 cpuload_end - cpuload_begin)

    def load_graph_into_gpu(self):
        """Pull whole graph from memory into gpu 
        """
        gpuload_begin = time.time()
        log.info("STAGE [GPU Load] begin load edges from cpu to gpu")
        for i in range(len(self.etype_list)):
            self.graph.upload_batch(0, i,
                                    len(get_cuda_places()), self.etype_list[i])
            log.info("STAGE [GPU Load] end load edge into GPU, type[" +
                     self.etype_list[i] + "]")

        slot_num = len(self.slots)
        log.info("STAGE [GPU Load] begin load node from cpu to gpu")
        if slot_num > 0:
            self.graph.upload_batch(1, len(get_cuda_places()), slot_num)
        log.info("STAGE [GPU Load] end load node from cpu to gpu")
        gpuload_end = time.time()
        log.info("STAGE [GPU LOAD] finished, time cost: %f sec",
                 gpuload_end - gpuload_begin)

    def finalize(self):
        """release the graph"""
        self.graph.finalize()

    def __del__(self):
        self.finalize()
