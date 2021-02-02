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
import os
import argparse
import time

import sys
import traceback
import numpy as np
import scipy.sparse as sp
from sklearn.preprocessing import StandardScaler

import pgl
from pgl.utils.logger import log
from pgl.utils import paddle_helper
import paddle
import paddle.fluid as fluid
import reader
from model import graphsage_mean, graphsage_meanpool,\
        graphsage_maxpool, graphsage_lstm


def load_data(normalize=True, symmetry=True):
    """
        data from https://github.com/matenure/FastGCN/issues/8
        reddit_adj.npz: https://drive.google.com/open?id=174vb0Ws7Vxk_QTUtxqTgDHSQ4El4qDHt
        reddit.npz: https://drive.google.com/open?id=19SphVl_Oe8SJ1r87Hr5a6znx3nJu1F2J
    """
    data_dir = os.path.dirname(os.path.abspath(__file__))
    data = np.load(os.path.join(data_dir, "data/reddit.npz"))
    adj = sp.load_npz(os.path.join(data_dir, "data/reddit_adj.npz"))
    if symmetry:
        adj = adj + adj.T
    adj = adj.tocoo()
    src = adj.row
    dst = adj.col

    num_class = 41

    train_label = data['y_train']
    val_label = data['y_val']
    test_label = data['y_test']

    train_index = data['train_index']
    val_index = data['val_index']
    test_index = data['test_index']

    feature = data["feats"].astype("float32")

    if normalize:
        scaler = StandardScaler()
        scaler.fit(feature[train_index])
        feature = scaler.transform(feature)

    log.info("Feature shape %s" % (repr(feature.shape)))
    graph = pgl.graph.Graph(
        num_nodes=feature.shape[0],
        edges=list(zip(src, dst)),
        node_feat={"feat": feature.astype("float32")})

    return {
        "graph": graph,
        "train_index": train_index,
        "train_label": train_label,
        "val_label": val_label,
        "val_index": val_index,
        "test_index": test_index,
        "test_label": test_label,
        "num_class": 41
    }


def build_graph_model(graph_wrapper, num_class, k_hop, graphsage_type,
                      hidden_size):
    """build_graph_model"""
    node_index = fluid.layers.data(
        "node_index", shape=[None], dtype="int64", append_batch_size=False)

    node_label = fluid.layers.data(
        "node_label", shape=[None, 1], dtype="int64", append_batch_size=False)

    feature = graph_wrapper.node_feat["feat"]

    for i in range(k_hop):
        if graphsage_type == 'graphsage_mean':
            feature = graphsage_mean(
                graph_wrapper,
                feature,
                hidden_size,
                act="relu",
                name="graphsage_mean_%s" % i)
        elif graphsage_type == 'graphsage_meanpool':
            feature = graphsage_meanpool(
                graph_wrapper,
                feature,
                hidden_size,
                act="relu",
                name="graphsage_meanpool_%s" % i)
        elif graphsage_type == 'graphsage_maxpool':
            feature = graphsage_maxpool(
                graph_wrapper,
                feature,
                hidden_size,
                act="relu",
                name="graphsage_maxpool_%s" % i)
        elif graphsage_type == 'graphsage_lstm':
            feature = graphsage_lstm(
                graph_wrapper,
                feature,
                hidden_size,
                act="relu",
                name="graphsage_maxpool_%s" % i)
        else:
            raise ValueError("graphsage type %s is not"
                             " implemented" % graphsage_type)

    feature = fluid.layers.gather(feature, node_index)
    logits = fluid.layers.fc(feature,
                             num_class,
                             act=None,
                             name='classification_layer')
    proba = fluid.layers.softmax(logits)

    loss = fluid.layers.softmax_with_cross_entropy(
        logits=logits, label=node_label)
    loss = fluid.layers.mean(loss)
    acc = fluid.layers.accuracy(input=proba, label=node_label, k=1)
    return loss, acc


def to_multidevice(batch_iter, num_trainer):
    """to_multidevice"""
    batch_dict = []
    for batch in batch_iter():
        batch_dict.append(batch)
        if len(batch_dict) == num_trainer:
            yield batch_dict
            batch_dict = []

    if len(batch_dict) > 0:
        log.warning("The batch (%s) can't fill all device (%s)"
                    "which will be discarded." %
                    (len(batch_dict), num_trainer))


def run_epoch(batch_iter,
              exe,
              program,
              prefix,
              model_loss,
              model_acc,
              epoch,
              log_per_step=100,
              num_trainer=1):
    """run_epoch"""
    batch = 0
    total_loss = 0.
    total_acc = 0.
    total_sample = 0
    start = time.time()
    if num_trainer > 1:
        batch_iter = to_multidevice(batch_iter, num_trainer)
    else:
        batch_iter = batch_iter()

    for batch_feed_dict in batch_iter:
        batch += 1
        if num_trainer > 1:
            batch_loss, batch_acc = exe.run(
                fetch_list=[model_loss.name, model_acc.name],
                feed=batch_feed_dict)

            batch_loss = np.mean(batch_loss)
            batch_acc = np.mean(batch_acc)
        else:
            batch_loss, batch_acc = exe.run(
                program,
                fetch_list=[model_loss.name, model_acc.name],
                feed=batch_feed_dict)

        if batch % log_per_step == 0:
            log.info("Batch %s %s-Loss %s %s-Acc %s" %
                     (batch, prefix, batch_loss, prefix, batch_acc))

        if num_trainer > 1:
            num_samples = sum(
                [len(_batch["node_index"]) for _batch in batch_feed_dict])
        else:
            num_samples = len(batch_feed_dict["node_index"])
        total_loss += batch_loss * num_samples
        total_acc += batch_acc * num_samples
        total_sample += num_samples
    end = time.time()

    log.info("%s Epoch %s Loss %.5lf Acc %.5lf Speed(per batch) %.5lf sec" %
             (prefix, epoch, total_loss / total_sample,
              total_acc / total_sample, (end - start) / batch))


def main(args):
    """main"""
    data = load_data(args.normalize, args.symmetry)
    log.info("preprocess finish")
    log.info("Train Examples: %s" % len(data["train_index"]))
    log.info("Val Examples: %s" % len(data["val_index"]))
    log.info("Test Examples: %s" % len(data["test_index"]))
    log.info("Num nodes %s" % data["graph"].num_nodes)
    log.info("Num edges %s" % data["graph"].num_edges)
    log.info("Average Degree %s" % np.mean(data["graph"].indegree()))

    place = fluid.CUDAPlace(0) if args.use_cuda else fluid.CPUPlace()
    train_program = fluid.Program()
    startup_program = fluid.Program()
    samples = []
    if args.samples_1 > 0:
        samples.append(args.samples_1)
    if args.samples_2 > 0:
        samples.append(args.samples_2)

    with fluid.program_guard(train_program, startup_program):
        graph_wrapper = pgl.graph_wrapper.GraphWrapper(
            "sub_graph",
            node_feat=data['graph'].node_feat_info())

        model_loss, model_acc = build_graph_model(
            graph_wrapper,
            num_class=data["num_class"],
            hidden_size=args.hidden_size,
            graphsage_type=args.graphsage_type,
            k_hop=len(samples))

    test_program = train_program.clone(for_test=True)

    with fluid.program_guard(train_program, startup_program):
        adam = fluid.optimizer.Adam(learning_rate=args.lr)
        adam.minimize(model_loss)

    exe = fluid.Executor(place)
    exe.run(startup_program)
    if args.num_trainer > 1:
        build_strategy = fluid.BuildStrategy()
        build_strategy.remove_unnecessary_lock = False
        build_strategy.enable_sequential_execution = True

        train_exe = fluid.ParallelExecutor(
            use_cuda=args.use_cuda,
            main_program=train_program,
            build_strategy=build_strategy,
            loss_name=model_loss.name)
    else:
        train_exe = exe

    train_iter = reader.multiprocess_graph_reader(
        data['graph'],
        graph_wrapper,
        samples=samples,
        num_workers=args.sample_workers,
        batch_size=args.batch_size,
        node_index=data['train_index'],
        node_label=data["train_label"])

    val_iter = reader.multiprocess_graph_reader(
        data['graph'],
        graph_wrapper,
        samples=samples,
        num_workers=args.sample_workers,
        batch_size=args.batch_size,
        node_index=data['val_index'],
        node_label=data["val_label"])

    test_iter = reader.multiprocess_graph_reader(
        data['graph'],
        graph_wrapper,
        samples=samples,
        num_workers=args.sample_workers,
        batch_size=args.batch_size,
        node_index=data['test_index'],
        node_label=data["test_label"])

    for epoch in range(args.epoch):
        run_epoch(
            train_iter,
            program=train_program,
            exe=train_exe,
            prefix="train",
            model_loss=model_loss,
            model_acc=model_acc,
            num_trainer=args.num_trainer,
            epoch=epoch)

        run_epoch(
            val_iter,
            program=test_program,
            exe=exe,
            prefix="val",
            model_loss=model_loss,
            model_acc=model_acc,
            log_per_step=10000,
            epoch=epoch)

    run_epoch(
        test_iter,
        program=test_program,
        prefix="test",
        exe=exe,
        model_loss=model_loss,
        model_acc=model_acc,
        log_per_step=10000,
        epoch=epoch)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='graphsage')
    parser.add_argument("--use_cuda", action='store_true', help="use_cuda")
    parser.add_argument(
        "--normalize", action='store_true', help="normalize features")
    parser.add_argument(
        "--symmetry", action='store_true', help="undirect graph")
    parser.add_argument("--graphsage_type", type=str, default="graphsage_mean")
    parser.add_argument("--sample_workers", type=int, default=5)
    parser.add_argument("--epoch", type=int, default=10)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_trainer", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--samples_1", type=int, default=25)
    parser.add_argument("--samples_2", type=int, default=10)
    args = parser.parse_args()
    log.info(args)
    main(args)
