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
"""train.py
"""
import argparse
import time
import glob
import os

import numpy as np

import pgl
from pgl.utils.logger import log
from pgl.utils import paddle_helper
import paddle
import paddle.fluid as fluid
import tqdm

import reader
import model


def get_layer(layer_type, gw, feature, hidden_size, act, name, is_test=False):
    """get_layer"""
    return getattr(model, layer_type)(gw, feature, hidden_size, act, name)


def load_pos_neg(data_path):
    """load_pos_neg"""
    train_eid = []
    train_src = []
    train_dst = []
    with open(data_path) as f:
        eid = 0
        for idx, line in tqdm.tqdm(enumerate(f)):
            src, dst = line.strip().split('\t')
            train_src.append(int(src))
            train_dst.append(int(dst))
            train_eid.append(int(eid))
            eid += 1
    # concate the the pos data and neg data
    train_eid = np.array(train_eid, dtype="int64")
    train_src = np.array(train_src, dtype="int64")
    train_dst = np.array(train_dst, dtype="int64")

    returns = {"train_data": (train_src, train_dst, train_eid), }
    return returns


def binary_op(u_embed, v_embed, binary_op_type):
    """binary_op"""
    if binary_op_type == "Average":
        edge_embed = (u_embed + v_embed) / 2
    elif binary_op_type == "Hadamard":
        edge_embed = u_embed * v_embed
    elif binary_op_type == "Weighted-L1":
        edge_embed = fluid.layers.abs(u_embed - v_embed)
    elif binary_op_type == "Weighted-L2":
        edge_embed = (u_embed - v_embed) * (u_embed - v_embed)
    else:
        raise ValueError(binary_op_type + " binary_op_type doesn't exists")
    return edge_embed


class RetDict(object):
    """RetDict"""
    pass


def build_graph_model(args):
    """build_graph_model"""
    node_feature_info = [('index', [None], np.dtype('int64'))]

    place = fluid.CUDAPlace(0) if args.use_cuda else fluid.CPUPlace()
    graph_wrappers = []
    feed_list = []

    graph_wrappers.append(
        pgl.graph_wrapper.GraphWrapper(
            "layer_0", node_feat=node_feature_info))
    #edge_feat=[("f", [None, 1], "float32")]))

    num_embed = args.num_nodes

    num_layers = args.num_layers

    src_index = fluid.layers.data(
        "src_index", shape=[None], dtype="int64", append_batch_size=False)

    dst_index = fluid.layers.data(
        "dst_index", shape=[None], dtype="int64", append_batch_size=False)

    feature = fluid.layers.embedding(
        input=fluid.layers.reshape(graph_wrappers[0].node_feat['index'],
                                   [-1, 1]),
        size=(num_embed + 1, args.hidden_size),
        is_sparse=args.is_sparse,
        is_distributed=args.is_distributed)

    features = [feature]
    ret_dict = RetDict()
    ret_dict.graph_wrappers = graph_wrappers
    edge_data = [src_index, dst_index]
    feed_list.extend(edge_data)
    ret_dict.feed_list = feed_list

    for i in range(num_layers):
        if i == num_layers - 1:
            act = None
        else:
            act = "leaky_relu"
        feature = get_layer(
            args.layer_type,
            graph_wrappers[0],
            feature,
            args.hidden_size,
            act,
            name="%s_%s" % (args.layer_type, i))
        features.append(feature)

    src_feat = fluid.layers.gather(features[-1], src_index)
    src_feat = fluid.layers.fc(src_feat,
                               args.hidden_size,
                               bias_attr=None,
                               param_attr=fluid.ParamAttr(name="feat"))
    dst_feat = fluid.layers.gather(features[-1], dst_index)
    dst_feat = fluid.layers.fc(dst_feat,
                               args.hidden_size,
                               bias_attr=None,
                               param_attr=fluid.ParamAttr(name="feat"))
    if args.phase == "predict":
        node_id = fluid.layers.data(
            "node_id", shape=[None, 1], dtype="int64", append_batch_size=False)
        ret_dict.src_feat = src_feat
        ret_dict.dst_feat = dst_feat
        ret_dict.id = node_id
        return ret_dict

    batch_size = args.batch_size
    batch_negative_label = fluid.layers.reshape(
        fluid.layers.range(0, batch_size, 1, "int64"), [-1, 1])
    batch_negative_label = fluid.layers.one_hot(batch_negative_label,
                                                batch_size)
    batch_loss_weight = (batch_negative_label *
                         (batch_size - 2) + 1.0) / (batch_size - 1)
    batch_loss_weight.stop_gradient = True
    batch_negative_label = batch_negative_label
    batch_negative_label = fluid.layers.cast(
        batch_negative_label, dtype="float32")
    batch_negative_label.stop_gradient = True

    cos_theta = fluid.layers.matmul(src_feat, dst_feat, transpose_y=True)

    # Calc Loss
    loss = fluid.layers.sigmoid_cross_entropy_with_logits(
        x=cos_theta, label=batch_negative_label)
    loss = loss * batch_loss_weight
    #loss = fluid.layers.reduce_sum(loss, -1)
    loss = fluid.layers.mean(loss)

    # Calc AUC
    proba = fluid.layers.sigmoid(cos_theta)
    proba = fluid.layers.reshape(proba, [-1, 1])
    proba = fluid.layers.concat([proba * -1 + 1, proba], axis=1)
    gold_label = fluid.layers.reshape(batch_negative_label, [-1, 1])
    gold_label = fluid.layers.cast(gold_label, "int64")
    auc, batch_auc_out, [batch_stat_pos, batch_stat_neg, stat_pos, stat_neg] = \
         fluid.layers.auc(input=proba, label=gold_label, curve='ROC', )

    ret_dict.loss = loss
    ret_dict.auc = batch_auc_out
    return ret_dict


def run_epoch(
        py_reader,
        exe,
        program,
        prefix,
        model_dict,
        epoch,
        batch_size,
        log_per_step=100,
        save_per_step=10000, ):
    """run_epoch"""
    batch = 0
    start = time.time()

    batch_end = time.time()

    for batch_feed_dict in py_reader():
        if prefix == "train":
            if batch_feed_dict["src_index"].shape[0] != batch_size:
                log.warning(
                    'batch_feed_dict["src_index"].shape[0] != 1024, continue')
                continue
        batch_start = time.time()
        batch += 1
        batch_loss, batch_auc = exe.run(
            program,
            feed=batch_feed_dict,
            fetch_list=[model_dict.loss.name, model_dict.auc.name])

        batch_end = time.time()
        if batch % log_per_step == 0:
            log.info(
                "Batch %s %s-Loss %s \t %s-Auc  %s \t Speed(per batch) %.5lf sec"
                % (batch, prefix, np.mean(batch_loss), prefix,
                   np.mean(batch_auc), batch_end - batch_start))
        if batch != 0 and batch % save_per_step == 0:
            fluid.io.save_params(
                exe, dirname='checkpoint', main_program=program)
    fluid.io.save_params(exe, dirname='checkpoint', main_program=program)


def run_predict_epoch(py_reader,
                      exe,
                      program,
                      prefix,
                      model_dict,
                      num_nodes,
                      hidden_size,
                      log_per_step=100):
    """run_predict_epoch"""
    batch = 0
    start = time.time()
    #use the parallel executor to speed up
    batch_end = time.time()
    all_feat = np.zeros((num_nodes, hidden_size), dtype="float32")

    for batch_feed_dict in tqdm.tqdm(py_reader()):
        batch_start = time.time()
        batch += 1
        batch_src_feat, batch_id = exe.run(
            program,
            feed=batch_feed_dict,
            fetch_list=[model_dict.src_feat.name, model_dict.id.name])

        for ind, id in enumerate(batch_id):
            all_feat[id] = batch_src_feat[ind]
    np.save("emb.npy", all_feat)


def main(args):
    """main"""
    place = fluid.CUDAPlace(0) if args.use_cuda else fluid.CPUPlace()
    exe = fluid.Executor(place)

    train_program = fluid.Program()
    startup_program = fluid.Program()

    with fluid.program_guard(train_program, startup_program):
        ret_dict = build_graph_model(args=args)

    val_program = train_program.clone(for_test=True)
    if args.phase == "train":
        with fluid.program_guard(train_program, startup_program):
            adam = fluid.optimizer.Adam(learning_rate=args.lr)
            adam.minimize(ret_dict.loss)
    # reset the place according to role of parameter server
    exe.run(startup_program)

    with open(args.data_path) as f:
        log.info("Begin Load Graph")
        src = []
        dst = []
        for idx, line in tqdm.tqdm(enumerate(f)):
            s, d = line.strip().split()
            src.append(s)
            dst.append(d)
            dst.append(s)
            src.append(d)
    src = np.array(src, dtype="int64").reshape(-1, 1)
    dst = np.array(dst, dtype="int64").reshape(-1, 1)
    edges = np.hstack([src, dst])

    log.info("Begin Build Index")
    ret_dict.graph = pgl.graph.Graph(num_nodes=args.num_nodes, edges=edges)
    ret_dict.graph.indegree()
    log.info("End Build Index")

    if args.phase == "train":
        #just the worker, load the sample
        data = load_pos_neg(args.data_path)

        feed_name_list = [var.name for var in ret_dict.feed_list]
        train_iter = reader.graph_reader(
            args.num_layers,
            ret_dict.graph_wrappers,
            batch_size=args.batch_size,
            data=data['train_data'],
            samples=args.samples,
            num_workers=args.sample_workers,
            feed_name_list=feed_name_list,
            use_pyreader=args.use_pyreader,
            graph=ret_dict.graph)

        # get PyReader 
        for epoch in range(args.epoch):
            epoch_start = time.time()
            try:
                run_epoch(
                    train_iter,
                    program=train_program,
                    exe=exe,
                    prefix="train",
                    model_dict=ret_dict,
                    epoch=epoch,
                    batch_size=args.batch_size,
                    log_per_step=1)
                epoch_end = time.time()
                print("Epoch: {0}, Train total expend: {1} ".format(
                    epoch, epoch_end - epoch_start))
            except Exception as e:
                log.info("Run Epoch Error %s" % e)
            fluid.io.save_params(
                exe,
                dirname=args.checkpoint + '_%s' % (epoch + 1),
                main_program=train_program)

            log.info("EPOCH END")

        log.info("RUN FINISH")
    elif args.phase == "predict":
        fluid.io.load_params(
            exe,
            dirname=args.checkpoint + '_%s' % args.epoch,
            main_program=val_program)
        test_src = np.arange(0, args.num_nodes, dtype="int64")
        feed_name_list = [var.name for var in ret_dict.feed_list]
        predict_iter = reader.graph_reader(
            args.num_layers,
            ret_dict.graph_wrappers,
            batch_size=args.batch_size,
            data=(test_src, test_src, test_src),
            samples=args.samples,
            num_workers=args.sample_workers,
            feed_name_list=feed_name_list,
            use_pyreader=args.use_pyreader,
            graph=ret_dict.graph,
            predict=True)
        run_predict_epoch(
            predict_iter,
            program=val_program,
            exe=exe,
            prefix="predict",
            hidden_size=args.hidden_size,
            model_dict=ret_dict,
            num_nodes=args.num_nodes,
            log_per_step=100)
        log.info("EPOCH END")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='graphsage')
    parser.add_argument(
        "--use_cuda", action='store_true', help="use_cuda", default=False)
    parser.add_argument("--layer_type", type=str, default="graphsage_mean")
    parser.add_argument("--epoch", type=int, default=1)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default="model_ckpt")
    parser.add_argument("--cache_path", type=str, default="./tmp")
    parser.add_argument("--phase", type=str, default="train")
    parser.add_argument("--digraph", action='store_true', default=False)
    parser.add_argument('--samples', nargs='+', type=int, default=[10, 10])
    parser.add_argument("--sample_workers", type=int, default=10)
    parser.add_argument("--num_nodes", type=int, required=True)
    parser.add_argument("--is_sparse", action='store_true', default=False)
    parser.add_argument("--is_distributed", action='store_true', default=False)
    parser.add_argument("--real_graph", action='store_true', default=True)
    parser.add_argument("--use_pyreader", action='store_true', default=False)
    args = parser.parse_args()
    log.info(args)
    main(args)
