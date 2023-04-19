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

import paddle
import paddle.fluid as fluid
import paddle.static as static
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
import util_hadoop as HFS


def train(args, exe, model_dict, dataset):
    """ training """
    train_msg = ""

    train_begin_time = time.time()
    dataset.dist_graph.load_train_node_from_file(args.train_start_nodes)
    for epoch in range(1, args.epochs + 1):
        if args.max_steps > 0 and model_util.print_count >= args.max_steps:
            log.info("training reach max_steps: %d, training end" %
                     args.max_steps)
            break

        epoch_begin = time.time()
        epoch_loss = 0
        train_pass_num = 0
        for pass_dataset in dataset.pass_generator(epoch):
            exe.train_from_dataset(
                model_dict.train_program, pass_dataset, debug=False)

            t_loss = util.get_global_value(model_dict.visualize_loss,
                                           model_dict.batch_count)
            epoch_loss += t_loss
            train_pass_num += 1

        epoch_end = time.time()
        log.info("epoch[%d] finished, time cost: %f sec" %
                 (epoch, epoch_end - epoch_begin))

        if train_pass_num > 0:
            epoch_loss = epoch_loss / train_pass_num
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

        is_save = (epoch % args.save_model_interval == 0 or
                   epoch == args.epochs)
        if args.model_save_path and is_save:
            savemodel_begin = time.time()
            log.info("saving model for epoch {}".format(epoch))
            dataset.embedding.dump_to_mem()
            ret = util.save_model(exe, model_dict, args, args.local_model_path,
                            args.model_save_path)
            fleet.barrier_worker()
            if ret != 0:
                log.warning("Fail to save model")
                return -1
            savemodel_end = time.time()
            log.info("STAGE [SAVE MODEL] for epoch [%d] finished, time cost: %f sec" \
                % (epoch, savemodel_end - savemodel_begin))

    train_end_time = time.time()
    log.info("STAGE [TRAIN MODEL] finished, time cost: % sec" %
             (train_end_time - train_begin_time))

    return 0


def train_with_multi_metapath(args, exe, model_dict, dataset):
    """ training with multiple metapaths """
    train_msg = ""

    sorted_metapaths, metapath_dict = \
        dataset.dist_graph.get_sorted_metapath_and_dict(args.meta_path)

    train_begin_time = time.time()
    for epoch in range(1, args.epochs + 1):
        epoch_begin = time.time()
        epoch_loss = 0
        train_pass_num = 0
        meta_path_len = len(sorted_metapaths)
        for i in range(meta_path_len):
            dataset.dist_graph.load_metapath_edges_nodes(
                metapath_dict, sorted_metapaths[i], i)
            metapath_train_begin = time.time()
            dataset.dist_graph.load_train_node_from_file(
                args.train_start_nodes)
            for pass_dataset in dataset.pass_generator():
                exe.train_from_dataset(
                    model_dict.train_program, pass_dataset, debug=False)
                t_loss = util.get_global_value(model_dict.visualize_loss,
                                               model_dict.batch_count)
                epoch_loss += t_loss
                train_pass_num += 1
            metapath_train_end = time.time()
            log.info("metapath[%s] [%d/%d] trained, pass_num[%d] time: %s" %
                     (sorted_metapaths[i], i, meta_path_len, pass_id + 1,
                      metapath_train_end - metapath_train_begin))
            dataset.dist_graph.clear_metapath_state()

        epoch_end = time.time()
        log.info("epoch[%d] finished, time cost: %f sec" %
                 (epoch, epoch_end - epoch_begin))

        if train_pass_num > 0:
            epoch_loss = epoch_loss / train_pass_num

        fleet.barrier_worker()
        time_msg = "%s\n" % datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        train_msg += time_msg
        msg = "Train: Epoch %d | meta path: %s | batch_loss %.6f\n" % \
                (epoch, sorted_metapaths[i], epoch_loss)
        train_msg += msg
        log.info(msg)

        if fleet.worker_index() == 0:
            with open(os.path.join('./train_result.log'), 'a') as f:
                f.write(time_msg)
                f.write(msg)

        fleet.barrier_worker()

        is_save = (epoch % args.save_model_interval == 0 or
                   epoch == args.epochs)
        if args.model_save_path and is_save:
            savemodel_begin = time.time()
            log.info("saving model for epoch {}".format(epoch))
            dataset.embedding.dump_to_mem()
            ret = util.save_model(exe, model_dict, args, args.local_model_path,
                            args.model_save_path)
            fleet.barrier_worker()
            if ret != 0:
                log.warning("Fail to save model")
                return -1
            savemodel_end = time.time()
            log.info("STAGE [SAVE MODEL] for epoch [%d] finished, time cost: %f sec" \
                % (epoch, savemodel_end - savemodel_begin))

    train_end_time = time.time()
    log.info("STAGE [TRAIN MODEL] finished, time cost: % sec" %
             (train_end_time - train_begin_time))

    return 0

def infer(args, exe, infer_model_dict, dataset):
    infer_begin = time.time()

    if hasattr(dataset.embedding.parameter_server, "set_mode"):
        dataset.embedding.set_infer_mode(True)

    dataset.dist_graph.load_infer_node_from_file(args.infer_nodes)
    for pass_dataset in dataset.pass_generator():
        exe.train_from_dataset(
            infer_model_dict.train_program, pass_dataset, debug=False)

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

    slot_num_for_pull_feature = 1 if args.token_slot else 0
    slot_num_for_pull_feature += len(args.slots)
    embedding = DistEmbedding(
        slots=model_dict.total_gpups_slots,
        embedding_size=args.emb_size,
        slot_num_for_pull_feature=slot_num_for_pull_feature)

    dist_graph = DistGraph(
        root_dir=args.graph_data_local_path,
        node_types=args.ntype2files,
        edge_types=args.etype2files,
        symmetry=args.symmetry,
        slots=args.slots,
        token_slot=args.token_slot,
        slot_num_for_pull_feature=slot_num_for_pull_feature,
        num_parts=args.num_part,
        metapath_split_opt=args.metapath_split_opt,
        infer_nodes=args.infer_nodes,
        use_weight=args.weighted_sample or args.return_weight)

    dist_graph.load_edge()
    ret = dist_graph.load_node()
    if ret != 0:
        log.warning("Fail to load node")
        return -1

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
        dist_graph=dist_graph)

    ret = 0
    if args.need_train:
        if args.metapath_split_opt:
            ret = train_with_multi_metapath(args, exe, model_dict, train_dataset)
        else:
            ret = train(args, exe, model_dict, train_dataset)
    else:
        log.info("STAGE: need_train is %s, skip training process" %
                 args.need_train)
    if ret != 0:
        log.warning("Fail to train")
        return -1

    fleet.barrier_worker()
    if args.need_inference:
        infer_dataset = InferDataset(
            args.chunk_num,
            dataset_config=args,
            holder_list=infer_model_dict.holder_list,
            infer_model_dict=infer_model_dict,
            embedding=embedding,
            dist_graph=dist_graph)

        infer(args, exe, infer_model_dict, infer_dataset)
    else:
        log.info("STAGE: need_inference is %s, skip inference process" %
                 args.need_inference)

    if args.need_dump_walk is True:
        upload_dump_begin = time.time()
        util.upload_dump_walk(args, args.local_dump_path)
        upload_dump_end = time.time()
        log.info("STAGE [UPLOAD DUMP WALK] finished, time cost: %f sec" \
                % (upload_dump_end - upload_dump_begin))
    return 0


def main(args):
    """main"""
    device_ids = get_cuda_places()
    place = paddle.CUDAPlace(device_ids[0])
    exe = static.Executor(place)
    if paddle.distributed.get_world_size() > 1:
        fleet.init(is_collective=True)
    else:
        fleet.init()

    # multi node save model path add rank id
    if paddle.distributed.get_world_size() > 1:
        worker_id = ("%03d" % (fleet.worker_index()))
        args.local_model_path = os.path.join(args.local_model_path, worker_id)
        args.local_result_path = os.path.join(args.local_result_path,
                                              worker_id)
        args.local_dump_path = os.path.join(args.local_dump_path, worker_id)
        print("args=%s" % (args))
    need_inference = args.need_inference
    infer_model_dict = None
    startup_program = static.Program()
    train_program = static.Program()

    print("workid={}, work num={}, node num={}, worker_endpoints={}".format(
        fleet.worker_index(),
        fleet.worker_num(), fleet.node_num(), fleet.worker_endpoints(True)))

    with static.program_guard(train_program, startup_program):
        with paddle.utils.unique_name.guard():
            model_dict = getattr(M, args.model_type)(config=args)

    model_dict.startup_program = startup_program
    model_dict.train_program = train_program

    adam = paddle.optimizer.Adam(learning_rate=args.dense_lr)
    optimizer = fleet.distributed_optimizer(
        adam, strategy=get_strategy(args, model_dict))
    optimizer.minimize(model_dict.loss, model_dict.startup_program)
    make_distributed_train_program(args, model_dict)

    if need_inference:
        infer_startup_program = static.Program()
        infer_train_program = static.Program()

        with static.program_guard(infer_train_program, infer_startup_program):
            with paddle.utils.unique_name.guard():
                infer_model_dict = getattr(M, args.model_type)(config=args,
                                                               is_predict=True)

        infer_model_dict.startup_program = infer_startup_program
        infer_model_dict.train_program = infer_train_program

        fake_lr_infer = 0.00
        adam_infer = paddle.optimizer.Adam(learning_rate=fake_lr_infer)
        optimizer1 = fleet.distributed_optimizer(
            adam_infer, strategy=get_strategy(args, infer_model_dict))
        optimizer1.minimize(infer_model_dict.loss,
                            infer_model_dict.startup_program)
        make_distributed_infer_program(args, infer_model_dict)

    # init and run server or worker
    if fleet.is_server():
        log.info("before run server")
        fleet.init_server()
        fleet.run_server()

    elif fleet.is_worker():
        ret = run_worker(args, exe, model_dict, infer_model_dict)
        if ret != 0:
            exe.close()
            fleet.stop_worker()
            return -1
    exe.close()
    fleet.stop_worker()
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PGLBox')
    parser.add_argument("--config", type=str, default="./config.yaml")
    args = parser.parse_args()

    util.print_useful_info()
    config = prepare_config(args.config)
    config.local_model_path = "./model"
    config.local_result_path = "./embedding"

    if config.train_mode == "online_train":
        config.working_root = os.path.join(config.working_root,
                                           config.newest_time)

    config.model_save_path = os.path.join(config.working_root, "model")
    config.infer_result_path = os.path.join(config.working_root, 'embedding')
    config.dump_save_path = os.path.join(config.working_root, 'dump_walk')

    config.max_steps = config.max_steps if config.max_steps else 0
    config.metapath_split_opt = config.metapath_split_opt \
                                if config.metapath_split_opt else False
    config.weighted_sample = config.weighted_sample if config.weighted_sample else False
    config.return_weight = config.return_weight if config.return_weight else False

    # set hadoop global account
    if config.output_fs_naame or config.output_fs_ugi:
        hadoop_bin = "%s/bin/hadoop" % (os.getenv("HADOOP_HOME"))
        HFS.set_hadoop_account(hadoop_bin, config.output_fs_name,
                               config.output_fs_ugi)

    print("#===================PRETTY CONFIG============================#")
    pretty(config, indent=0)
    print("#===================PRETTY CONFIG============================#")
    ret = main(config)

    if config.train_mode == "online_train":
        # update warm_start_time for next timestamp training
        cmd = 'sed -i "s|^warm_start_from: .*$|warm_start_from: %s|" ./config.yaml' \
                % (config.model_save_path)
        util.run_cmd(cmd)

    exit(ret)
