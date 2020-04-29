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
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals
import argparse
import pickle
import time
import glob
import os
import io
import traceback
import pickle as pkl
role = os.getenv("TRAINING_ROLE", "TRAINER")

import numpy as np
import yaml
from easydict import EasyDict as edict
import pgl
from pgl.utils.logger import log
from pgl.utils import paddle_helper
import paddle
import paddle.fluid as F

from models.model_factory import Model
from dataset.graph_reader import GraphGenerator 


class PredictData(object):
    def __init__(self, num_nodes):
        trainer_id = int(os.getenv("PADDLE_TRAINER_ID", "0"))
        trainer_count = int(os.getenv("PADDLE_TRAINERS_NUM", "1"))
        train_usr = np.arange(trainer_id, num_nodes, trainer_count)
        #self.data = (train_usr, train_usr)
        self.data = train_usr

    def __getitem__(self, index):
        return [self.data[index], self.data[index]]

def tostr(data_array):
    return " ".join(["%.5lf" % d for d in  data_array])

def run_predict(py_reader,
              exe,
              program,
              model_dict,
              log_per_step=1,
              args=None):

    if args.input_type == "text":
        id2str = np.load(os.path.join(args.graph_path, "id2str.npy"), mmap_mode="r")

    trainer_id = int(os.getenv("PADDLE_TRAINER_ID", "0"))
    trainer_count = int(os.getenv("PADDLE_TRAINERS_NUM", "1"))
    if not os.path.exists(args.output_path):
        os.mkdir(args.output_path)

    fout = io.open("%s/part-%s" % (args.output_path, trainer_id), "w", encoding="utf8")
    batch = 0
        
    for batch_feed_dict in py_reader():
        batch += 1
        batch_usr_feat, batch_ad_feat, batch_src_real_index = exe.run(
            program,
            feed=batch_feed_dict,
            fetch_list=model_dict.outputs)

        if batch % log_per_step == 0:
            log.info("Predict %s finished" % batch)

        for ufs, _, sri in zip(batch_usr_feat, batch_ad_feat, batch_src_real_index):
            if args.input_type == "text":
                sri = id2str[int(sri)]
            line = "{}\t{}\n".format(sri, tostr(ufs))
            fout.write(line)

    fout.close()

def _warmstart(exe, program, path='params'):
    def _existed_persitables(var):
        #if not isinstance(var, fluid.framework.Parameter):
        #    return False
        if not F.io.is_persistable(var):
            return False
        param_path = os.path.join(path, var.name)
        log.info("Loading parameter: {} persistable: {} exists: {}".format(
            param_path,
            F.io.is_persistable(var),
            os.path.exists(param_path),
        ))
        return os.path.exists(param_path)
    F.io.load_vars(
        exe,
        path,
        main_program=program,
        predicate=_existed_persitables
    )

def main(config):
    model = Model.factory(config)

    if config.learner_type == "cpu":
        place = F.CPUPlace()
    elif config.learner_type == "gpu":
        gpu_id = int(os.getenv("FLAGS_selected_gpus", "0"))
        place = F.CUDAPlace(gpu_id)
    else:
        raise ValueError

    exe = F.Executor(place)

    val_program = F.default_main_program().clone(for_test=True)
    exe.run(F.default_startup_program()) 
    _warmstart(exe, F.default_startup_program(), path=config.infer_model)

    num_threads = int(os.getenv("CPU_NUM", 1))
    trainer_id = int(os.getenv("PADDLE_TRAINER_ID", 0))

    exec_strategy = F.ExecutionStrategy()
    exec_strategy.num_threads = num_threads
    build_strategy = F.BuildStrategy()
    build_strategy.enable_inplace = True
    build_strategy.memory_optimize = True
    build_strategy.remove_unnecessary_lock = False
    build_strategy.memory_optimize = False

    if num_threads > 1:
        build_strategy.reduce_strategy = F.BuildStrategy.ReduceStrategy.Reduce

    val_compiled_prog = F.compiler.CompiledProgram(
        val_program).with_data_parallel(
            build_strategy=build_strategy,
            exec_strategy=exec_strategy)

    num_nodes = int(np.load(os.path.join(config.graph_path, "num_nodes.npy")))

    predict_data = PredictData(num_nodes)

    predict_iter = GraphGenerator(
        graph_wrappers=model.graph_wrappers,
        batch_size=config.infer_batch_size,
        data=predict_data,
        samples=config.samples,
        num_workers=config.sample_workers,
        feed_name_list=[var.name for var in model.feed_list],
        use_pyreader=config.use_pyreader,
        phase="predict",
        graph_data_path=config.graph_path,
        shuffle=False)

    if config.learner_type == "cpu":
        model.data_loader.decorate_batch_generator(
            predict_iter, places=F.cpu_places())
    elif config.learner_type == "gpu":
        gpu_id = int(os.getenv("FLAGS_selected_gpus", "0"))
        place = F.CUDAPlace(gpu_id)
        model.data_loader.decorate_batch_generator(
            predict_iter, places=place)
    else:
        raise ValueError

    run_predict(model.data_loader,
                program=val_compiled_prog,
                exe=exe,
                model_dict=model,
                args=config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='main')
    parser.add_argument("--conf", type=str, default="./config.yaml")
    args = parser.parse_args()
    config = edict(yaml.load(open(args.conf), Loader=yaml.FullLoader))
    print(config)
    main(config)
