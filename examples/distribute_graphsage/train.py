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


def load_data():
    """
        data from https://github.com/matenure/FastGCN/issues/8
        reddit.npz: https://drive.google.com/open?id=19SphVl_Oe8SJ1r87Hr5a6znx3nJu1F2J
        reddit_index_label is preprocess from reddit.npz without feats key.
    """
    data_dir = os.path.dirname(os.path.abspath(__file__))
    data = np.load(os.path.join(data_dir, "data/reddit_index_label.npz"))

    num_class = 41

    train_label = data['y_train']
    val_label = data['y_val']
    test_label = data['y_test']

    train_index = data['train_index']
    val_index = data['val_index']
    test_index = data['test_index']

    return {
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
    node_index = fluid.layers.data(
        "node_index", shape=[None], dtype="int64", append_batch_size=False)

    node_label = fluid.layers.data(
        "node_label", shape=[None, 1], dtype="int64", append_batch_size=False)

    #feature = fluid.layers.gather(feature, graph_wrapper.node_feat['feats'])
    feature = graph_wrapper.node_feat['feats']
    feature.stop_gradient = True

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


def run_epoch(batch_iter,
              exe,
              program,
              prefix,
              model_loss,
              model_acc,
              epoch,
              log_per_step=100):
    batch = 0
    total_loss = 0.
    total_acc = 0.
    total_sample = 0
    start = time.time()
    for batch_feed_dict in batch_iter():
        batch += 1
        batch_loss, batch_acc = exe.run(program,
                                        fetch_list=[model_loss, model_acc],
                                        feed=batch_feed_dict)

        if batch % log_per_step == 0:
            log.info("Batch %s %s-Loss %s %s-Acc %s" %
                     (batch, prefix, batch_loss, prefix, batch_acc))

        num_samples = len(batch_feed_dict["node_index"])
        total_loss += batch_loss * num_samples
        total_acc += batch_acc * num_samples
        total_sample += num_samples
    end = time.time()

    log.info("%s Epoch %s Loss %.5lf Acc %.5lf Speed(per batch) %.5lf sec" %
             (prefix, epoch, total_loss / total_sample,
              total_acc / total_sample, (end - start) / batch))


def main(args):
    data = load_data()
    log.info("preprocess finish")
    log.info("Train Examples: %s" % len(data["train_index"]))
    log.info("Val Examples: %s" % len(data["val_index"]))
    log.info("Test Examples: %s" % len(data["test_index"]))

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
            fluid.CPUPlace(),
            node_feat=[('feats', [None, 602], np.dtype('float32'))])
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

    train_iter = reader.multiprocess_graph_reader(
        graph_wrapper,
        samples=samples,
        num_workers=args.sample_workers,
        batch_size=args.batch_size,
        node_index=data['train_index'],
        node_label=data["train_label"])

    val_iter = reader.multiprocess_graph_reader(
        graph_wrapper,
        samples=samples,
        num_workers=args.sample_workers,
        batch_size=args.batch_size,
        node_index=data['val_index'],
        node_label=data["val_label"])

    test_iter = reader.multiprocess_graph_reader(
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
            exe=exe,
            prefix="train",
            model_loss=model_loss,
            model_acc=model_acc,
            log_per_step=1,
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
    parser.add_argument("--sample_workers", type=int, default=10)
    parser.add_argument("--epoch", type=int, default=10)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--samples_1", type=int, default=25)
    parser.add_argument("--samples_2", type=int, default=10)
    args = parser.parse_args()
    log.info(args)
    main(args)
