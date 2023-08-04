# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved
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
"""doc
"""

import os
import sys
import time
import tqdm
import yaml
import argparse
import numpy as np

import pgl
from pgl.utils.logger import log

import paddle
import paddle.nn as nn
from paddle.optimizer import Adam
import paddle.distributed.fleet as fleet

from utils.config import prepare_config
from utils.logger import log_to_file
import models as M
from datasets.dist_dataloader import DistCPUDataloader
import datasets.dataset as DS
#  from utils.ps_util import DistributedInfer
from paddle.distributed.fleet.utils.ps_util import DistributedInfer

paddle.set_device("cpu")
paddle.enable_static()
fleet.init()


def main(config, ip_list_file, save_dir, infer_from):
    config.embed_type = "SparseEmbedding"
    log.info("building model with embed_type (%s)" % config.embed_type)
    model = getattr(M, config.model_type)(config, mode="distcpu")
    feed_dict, py_reader = model.get_static_input()
    pred = model(feed_dict)
    loss = model.loss(pred)

    place = paddle.CPUPlace()
    exe = paddle.static.Executor(place)

    dist_infer = DistributedInfer(
        main_program=paddle.static.default_main_program(),
        startup_program=paddle.static.default_startup_program())
    log.info("infer from %s" % infer_from)
    dist_infer.init_distributed_infer_env(exe, loss, dirname=infer_from)

    if fleet.is_worker():
        infer_program = dist_infer.get_dist_infer_program()

        ds = getattr(DS, config.dataset_type)(config,
                                              ip_list_file,
                                              mode="distcpu_infer")
        loader = DistCPUDataloader(
            ds,
            batch_size=config.batch_pair_size,
            num_workers=1,  # must be 1 in inference mode
            collate_fn=getattr(DS, config.collatefn)(config, mode="distcpu"))

        py_reader.set_batch_generator(lambda: loader)
        fleet.barrier_worker()
        log.info("begin inference in worker [%s]" % fleet.worker_index())
        inference(save_dir, exe, infer_program, py_reader, model)
        log.info("inference finished in worker [%s], waiting other workers..." \
                % (fleet.worker_index()))
        fleet.barrier_worker()
        log.info("stopping workers...")
        fleet.stop_worker()
        log.info("successfully stopping worker [%s]" % fleet.worker_index())


def inference(save_dir, exe, program, reader, model):
    save_file = os.path.join(save_dir, "part-%05d" % fleet.worker_index())
    log.info("node representations will be saved in %s" % save_file)

    node_ids, node_embed = model.get_embedding()

    reader.start()
    start = time.time()
    cc = 0
    with open(save_file, "w") as writer:
        try:
            while True:
                nids, nembed = exe.run(
                    program, fetch_list=[node_ids.name, node_embed.name])

                for nid, vec in zip(nids.reshape(-1), nembed):
                    str_vec = ' '.join(map(str, vec))
                    writer.write("%s\t%s\n" % (nid, str_vec))
                    cc += 1

                    if cc % 10000 == 0:
                        log.info("%s nodes have been processed" % cc)

        except paddle.framework.core.EOFException:
            reader.reset()
    log.info("total %s nodes have been processed" % cc)
    log.info("node representations are saved in %s" % save_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='GraphRec')
    parser.add_argument("--config", type=str, default="./config.yaml")
    parser.add_argument("--ip", type=str, default="./ip_list.txt")
    parser.add_argument("--task_name", type=str, default="graph_rec")
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--infer_from", type=str, default=None)
    args = parser.parse_args()

    if not os.path.exists(args.save_dir) and fleet.worker_index() == 0:
        os.makedirs(args.save_dir)

    config = prepare_config(args.config)
    main(config, args.ip, args.save_dir, args.infer_from)
