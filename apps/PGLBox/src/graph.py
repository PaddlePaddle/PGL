
import time
import helper
import util
from paddle.fluid.core import GraphGpuWrapper
from place import get_cuda_places
from pgl.utils.logger import log

class DistGraph(object):
    """ Initialize the Distributed Graph Server

    Args:
        root_dir: the graph data dir 
    
        node_types: the node type configs
 
        edge_types: the edge type configs.

        symmetry: whether the edges are symmetry

        slots: the node feature slot 

        num_parts: the sharded parts of graph data
    """
    def __init__(self, root_dir, node_types, edge_types, symmetry, slots, num_parts):
        self.node_types = node_types
        self.edge_types = edge_types
        self.slots = slots
        self.symmetry = symmetry
        self.root_dir = root_dir
        self.num_parts = num_parts

        etype2files = helper.parse_files(self.edge_types)
        self.etype_list = util.get_all_edge_type(etype2files, self.symmetry)

        ntype2files = helper.parse_files(self.node_types)
        self.ntype_list = list(ntype2files.keys())
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
                self.graph.add_table_feat_conf(ntype, str(slot_id), "feasign", 1)
        self.graph.set_slot_feature_separator(":")
        self.graph.set_feature_separator(",")
        self.graph.init_service()

    def load_graph_into_cpu(self):
        """Pull whole graph from disk into memory
        """
        cpuload_begin = time.time()
        reverse = 1 if self.symmetry else 0
        self.graph.load_node_and_edge(self.edge_types, self.node_types,
            self.root_dir, self.num_parts, reverse)
        cpuload_end = time.time()
        log.info("STAGE [CPU LOAD] finished, time cost: %f sec", cpuload_end - cpuload_begin)

    def load_graph_into_gpu(self):
        """Pull whole graph from memory into gpu 
        """
        gpuload_begin = time.time()
        log.info("STAGE [GPU Load] begin load edges from cpu to gpu")
        for i in range(len(self.etype_list)):
            self.graph.upload_batch(0, i, len(get_cuda_places()), self.etype_list[i])
            log.info("STAGE [GPU Load] end load edge into GPU, type[" + self.etype_list[i] + "]")

        slot_num = len(self.slots)
        log.info("STAGE [GPU Load] begin load node from cpu to gpu")
        if slot_num > 0:
            self.graph.upload_batch(1, len(get_cuda_places()), slot_num)
        log.info("STAGE [GPU Load] end load node from cpu to gpu")
        gpuload_end = time.time()
        log.info("STAGE [GPU LOAD] finished, time cost: %f sec", gpuload_end - gpuload_begin)

    def finalize(self):
        """release the graph"""
        self.graph.finalize()

    def __del__(self):
         self.finalize()
