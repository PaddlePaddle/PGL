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
"""Training Script for PGLBox
"""

import paddle
paddle.enable_static()

import os
import sys
import time
import glob
import yaml
import shutil
import argparse
import traceback
import pickle as pkl
import numpy as np
import helper
from datetime import datetime

import paddle.fluid as fluid
from place import get_cuda_places
from pgl.utils.logger import log
from datetime import datetime
from config_fleet import get_strategy
import util
import models as M

from paddle.distributed import fleet
from embedding import DistEmbedding
from graph import DistGraph
from dataset import UnsupReprLearningDataset, InferDataset
from distributed_program import make_distributed_train_program, make_distributed_infer_program
from util_config import prepare_config, pretty


def train(args, exe, model_dict, dataset):
    """ training """
    train_msg = ""

    for epoch in range(1, args.epochs + 1):
        epoch_loss = 0
        for chunk_dataset in dataset.chunk_generator():
            train_begin = time.time()
            exe.train_from_dataset(
                model_dict.train_program, chunk_dataset, debug=False)

            train_end = time.time()
            log.info("STAGE [SAMPLE AND TRAIN] for epoch [%d] finished, time cost: %f sec" \
                 % (epoch, train_end - train_begin))

            t_loss = util.get_global_value(model_dict.visualize_loss,
                                           model_dict.batch_count)
            epoch_loss += t_loss

        epoch_loss = epoch_loss / args.chunk_num
        fleet.barrier_worker()
        time_msg = "%s\n" % datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        train_msg += time_msg
        msg = "Train: Epoch %s | batch_loss %.6f\n" % (epoch, epoch_loss)
        train_msg += msg
        log.info(msg)

        if fleet.worker_index() == 0:
            with open(os.path.join('./train_result.log'), 'a') as f:
                f.write(time_msg)
                f.write(msg)

        fleet.barrier_worker()

        savemodel_begin = time.time()
        is_save = (epoch % args.save_model_interval == 0 or
                   epoch == args.epochs)
        if args.model_save_path and is_save:
            log.info("save model for epoch {}".format(epoch))
            util.save_model(exe, model_dict, args, args.local_model_path,
                            args.model_save_path)
        fleet.barrier_worker()
        savemodel_end = time.time()
        log.info("STAGE [SAVE MODEL] for epoch [%d] finished, time cost: %f sec" \
            % (epoch, savemodel_end - savemodel_begin))


def infer(args, exe, infer_model_dict, dataset):
    infer_begin = time.time()

    for chunk_dataset in dataset.chunk_generator():
        exe.train_from_dataset(
            infer_model_dict.train_program, chunk_dataset, debug=False)
    util.upload_embedding(args, args.local_result_path)
    infer_end = time.time()
    log.info("STAGE [INFER MODEL] finished, time cost: % sec" %
             (infer_end - infer_begin))


def run_worker(args, exe, model_dict, infer_model_dict):
    """
    run worker
    """
    need_inference = args.need_inference
    exe.run(model_dict.startup_program)
    if need_inference:
        exe.run(infer_model_dict.startup_program)
    fleet.init_worker()

    log.info("gpu ps slot names: %s" % repr(model_dict.total_gpups_slots))

    embedding = DistEmbedding(
        slots=model_dict.total_gpups_slots, embedding_size=args.emb_size)

    graph = DistGraph(
        root_dir=args.graph_data_local_path,
        node_types=args.ntype2files,
        edge_types=args.etype2files,
        symmetry=args.symmetry,
        slots=args.slots,
        num_parts=args.num_part)

    graph.load_graph_into_cpu()

    graph.load_graph_into_gpu()

    if args.warm_start_from:
        log.info("warmup start from %s" % args.warm_start_from)
        load_model_begin = time.time()
        util.load_pretrained_model(exe, model_dict, args, args.warm_start_from)
        load_model_end = time.time()
        log.info("STAGE [LOAD MODEL] finished, time cost: %f sec" \
            % (load_model_end - load_model_begin))

    fleet.barrier_worker()

    train_dataset = UnsupReprLearningDataset(
        args.chunk_num,
        dataset_config=args,
        holder_list=model_dict.holder_list,
        embedding=embedding,
        graph=graph)

    infer_dataset = InferDataset(
        args.chunk_num,
        dataset_config=args,
        holder_list=model_dict.holder_list,
        embedding=embedding,
        graph=graph)

    if args.need_train:
        train(args, exe, model_dict, train_dataset)
    else:
        log.info("STAGE: need_train is %s, skip training process" %
                 args.need_train)

    fleet.barrier_worker()
    if args.need_inference:
        infer(args, exe, infer_model_dict, infer_dataset)
    else:
        log.info("STAGE: need_inference is %s, skip inference process" %
                 args.need_inference)


def main(args):
    """main"""
    device_ids = get_cuda_places()
    place = fluid.CUDAPlace(device_ids[0])
    exe = fluid.Executor(place)
    fleet.init()
    need_inference = args.need_inference
    infer_model_dict = None
    startup_program = fluid.Program()
    train_program = fluid.Program()

    with fluid.program_guard(train_program, startup_program):
        with fluid.unique_name.guard():
            model_dict = getattr(M, args.model_type)(config=args)

    model_dict.startup_program = startup_program
    model_dict.train_program = train_program

    adam = fluid.optimizer.Adam(learning_rate=args.dense_lr)
    optimizer = fleet.distributed_optimizer(
        adam, strategy=get_strategy(args, model_dict))
    optimizer.minimize(model_dict.loss, model_dict.startup_program)
    make_distributed_train_program(args, model_dict)

    if need_inference:
        infer_startup_program = fluid.Program()
        infer_train_program = fluid.Program()

        with fluid.program_guard(infer_train_program, infer_startup_program):
            with fluid.unique_name.guard():
                infer_model_dict = getattr(M, args.model_type)(config=args,
                                                               is_predict=True)

        infer_model_dict.startup_program = infer_startup_program
        infer_model_dict.train_program = infer_train_program

        fake_lr_infer = 0.00
        adam_infer = fluid.optimizer.Adam(learning_rate=fake_lr_infer)
        optimizer1 = fleet.distributed_optimizer(
            adam_infer, strategy=get_strategy(args, infer_model_dict))
        optimizer1.minimize(infer_model_dict.loss,
                            infer_model_dict.startup_program)
        make_distributed_infer_program(args, infer_model_dict)

    # init and run server or worker
    if fleet.is_server():
        fleet.init_server()
        fleet.run_server()

    elif fleet.is_worker():
        run_worker(args, exe, model_dict, infer_model_dict)
        #pass

    fleet.stop_worker()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PGLBox')
    parser.add_argument("--config", type=str, default="./config.yaml")
    args = parser.parse_args()

    util.print_useful_info()
    config = prepare_config(args.config)
    config.local_model_path = "./model"
    config.local_result_path = "./embedding"
    config.model_save_path = os.path.join(config.working_root, "model")
    config.infer_result_path = os.path.join(config.working_root, 'embedding')
    print("#===================PRETTY CONFIG============================#")
    pretty(config, indent=0)
    print("#===================PRETTY CONFIG============================#")
    main(config)
