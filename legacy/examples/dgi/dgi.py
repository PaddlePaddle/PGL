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
"""
    DGI Pretrain
"""
import os
import pgl
from pgl import data_loader
from pgl.utils.logger import log
import paddle.fluid as fluid
import numpy as np
import time
import argparse


def load(name):
    """Load dataset"""
    if name == 'cora':
        dataset = data_loader.CoraDataset()
    elif name == "pubmed":
        dataset = data_loader.CitationDataset("pubmed", symmetry_edges=False)
    elif name == "citeseer":
        dataset = data_loader.CitationDataset("citeseer", symmetry_edges=False)
    else:
        raise ValueError(name + " dataset doesn't exists")
    return dataset


def save_param(dirname, var_name_list):
    """save_param"""
    for var_name in var_name_list:
        var = fluid.global_scope().find_var(var_name)
        var_tensor = var.get_tensor()
        np.save(os.path.join(dirname, var_name + '.npy'), np.array(var_tensor))


def main(args):
    """main"""
    dataset = load(args.dataset)

    # normalize
    indegree = dataset.graph.indegree()
    norm = np.zeros_like(indegree, dtype="float32")
    norm[indegree > 0] = np.power(indegree[indegree > 0], -0.5)
    dataset.graph.node_feat["norm"] = np.expand_dims(norm, -1)

    place = fluid.CUDAPlace(0) if args.use_cuda else fluid.CPUPlace()
    train_program = fluid.Program()
    startup_program = fluid.Program()
    hidden_size = 512

    with fluid.program_guard(train_program, startup_program):
        pos_gw = pgl.graph_wrapper.GraphWrapper(
            name="pos_graph",
            place=place,
            node_feat=dataset.graph.node_feat_info())

        neg_gw = pgl.graph_wrapper.GraphWrapper(
            name="neg_graph",
            place=place,
            node_feat=dataset.graph.node_feat_info())

        positive_feat = pgl.layers.gcn(pos_gw,
                                       pos_gw.node_feat["words"],
                                       hidden_size,
                                       activation="relu",
                                       norm=pos_gw.node_feat['norm'],
                                       name="gcn_layer_1")

        negative_feat = pgl.layers.gcn(neg_gw,
                                       neg_gw.node_feat["words"],
                                       hidden_size,
                                       activation="relu",
                                       norm=neg_gw.node_feat['norm'],
                                       name="gcn_layer_1")

        summary_feat = fluid.layers.sigmoid(
            fluid.layers.reduce_mean(
                positive_feat, [0], keep_dim=True))

        summary_feat = fluid.layers.fc(summary_feat,
                                       hidden_size,
                                       bias_attr=False,
                                       name="discriminator")
        pos_logits = fluid.layers.matmul(
            positive_feat, summary_feat, transpose_y=True)
        neg_logits = fluid.layers.matmul(
            negative_feat, summary_feat, transpose_y=True)
        pos_loss = fluid.layers.sigmoid_cross_entropy_with_logits(
            x=pos_logits,
            label=fluid.layers.ones(
                shape=[dataset.graph.num_nodes, 1], dtype="float32"))
        neg_loss = fluid.layers.sigmoid_cross_entropy_with_logits(
            x=neg_logits,
            label=fluid.layers.zeros(
                shape=[dataset.graph.num_nodes, 1], dtype="float32"))
        loss = fluid.layers.reduce_mean(pos_loss) + fluid.layers.reduce_mean(
            neg_loss)

        adam = fluid.optimizer.Adam(learning_rate=1e-3)
        adam.minimize(loss)

    exe = fluid.Executor(place)
    exe.run(startup_program)

    best_loss = 1e9
    dur = []

    for epoch in range(args.epoch):
        feed_dict = pos_gw.to_feed(dataset.graph)
        node_feat = dataset.graph.node_feat["words"].copy()
        perm = np.arange(0, dataset.graph.num_nodes)
        np.random.shuffle(perm)

        dataset.graph.node_feat["words"] = dataset.graph.node_feat["words"][
            perm]

        feed_dict.update(neg_gw.to_feed(dataset.graph))
        dataset.graph.node_feat["words"] = node_feat
        if epoch >= 3:
            t0 = time.time()
        train_loss = exe.run(train_program,
                             feed=feed_dict,
                             fetch_list=[loss],
                             return_numpy=True)
        if train_loss[0] < best_loss:
            best_loss = train_loss[0]
            save_param(args.checkpoint, ["gcn_layer_1", "gcn_layer_1_bias"])

        if epoch >= 3:
            time_per_epoch = 1.0 * (time.time() - t0)
            dur.append(time_per_epoch)

        log.info("Epoch %d " % epoch + "(%.5lf sec) " % np.mean(dur) +
                 "Train Loss: %f " % train_loss[0])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DGI pretrain')
    parser.add_argument(
        "--dataset", type=str, default="cora", help="dataset (cora, pubmed)")
    parser.add_argument(
        "--checkpoint", type=str, default="best_model", help="checkpoint")
    parser.add_argument(
        "--epoch", type=int, default=200, help="pretrain epochs")
    parser.add_argument("--use_cuda", action='store_true', help="use_cuda")
    args = parser.parse_args()
    log.info(args)
    main(args)
