"""Ernie
"""
from models.base  import BaseNet, BaseGNNModel 

class Ernie(BaseNet):

    def build_inputs(self):
        inputs = super(Ernie, self).build_inputs()
        term_ids = L.data(
            "term_ids", shape=[None, self.config.max_seqlen], dtype="int64", append_batch_size=False)
        return inputs + [term_ids]

    def build_embedding(self, graph_wrappers, term_ids):
        term_ids = L.unsqueeze(term_ids, [-1])
        ernie_config = self.config.ernie_config
        ernie = ErnieModel(
            src_ids=term_ids,
            sentence_ids=L.zeros_like(term_ids),
            task_ids=None,
            config=ernie_config,
            use_fp16=False,
            name="student_")
        feature = ernie.get_pooled_output()
        return feature

    def __call__(self, graph_wrappers):
        inputs = self.build_inputs()
        feature = self.build_embedding(graph_wrappers, inputs[-1])
        features = [feature]
        outputs = [self.take_final_feature(features[-1], i, "final_fc") for i in inputs[:-1]]
        src_real_index = L.gather(graph_wrappers[0].node_feat['index'], inputs[0])
        outputs.append(src_real_index)
        return inputs, outputs


class ErnieModel(BaseGNNModel):
    def gen_net_fn(self, config):
        return Ernie(config)


