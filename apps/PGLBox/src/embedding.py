import paddle
import paddle.fluid.core as core
from pgl.utils.logger import log
from place import get_cuda_places

class DistEmbedding(object):
    """ Setting the Embedding for the parameter server

    Args:

        slots: a list of int represents the slot key

        embedding_size: the output size of the embedding.
    """
    def __init__(self, slots, embedding_size):
        self.parameter_server = core.PSGPU()
        self.parameter_server.set_slot_vector(slots)
        self.parameter_server.init_gpu_ps(get_cuda_places())
        self.parameter_server.set_slot_dim_vector([embedding_size] * len(slots))

    def finalize(self):
        self.parameter_server.finalize()

    def begin_pass(self):
        self.parameter_server.begin_pass()

    def end_pass(self):
        self.parameter_server.end_pass()

    def __del__(self):
        self.finalize()

