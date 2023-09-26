# Copyright (c) 2021 PaddlePaddle/PGL Authors. All Rights Reserved.
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
import math
import random
import numpy as np
import paddle
from paddle.io import DataLoader
import pgl
from pgl.utils.logger import log
from data import EdgeDataset, StrucVecGraph, MLPDataset
from model import MLPModel, CECrition


def learning_embedding_from_struc2vec(args):
    """
    Learning the word2vec from the random path
    """
    from gensim.models import Word2Vec
    from gensim.models.word2vec import LineSentence
    struc_walks = LineSentence(args.tag + "_walk_path")
    model = Word2Vec(struc_walks, size=args.w2v_emb_size, window=args.w2v_window_size, iter=args.w2v_epoch, \
        min_count=0, hs=1, sg=1, workers=5)
    model.wv.save_word2vec_format(args.emb_file)


def generate_samples(args):
    """
    Generate the samples from the embedding, use the struc2vec emebedding as feature
    """
    if not os.path.exists(args.emb_file):
        raise Exception(
            "The embedding file is not exist, please generate the emebdding.")
    if not os.path.exists(args.label_file + "_reindex"):
        raise Exception(
            "The label index file is not exist, please generate the embedding.")
    emb_file = open(args.emb_file)
    file_label_reindex = open(args.label_file + "_reindex")
    label_dict = dict()
    # Convert the node to id 
    for line in file_label_reindex:
        items = line.strip("\n\r").split(" ")
        try:
            label_dict[int(items[0])] = int(items[1])
        except:
            continue

    data_for_train_valid = []
    for line in emb_file:
        items = line.strip("\n\r").split(" ")
        if len(items) <= 2:
            continue
        index = int(items[0])
        label = int(label_dict[index])
        sample = []
        sample.append(index)
        feature_emb = items[1:]
        feature_emb = [float(feature) for feature in feature_emb]
        sample.extend(feature_emb)
        sample.append(label)
        data_for_train_valid.append(sample)
    return data_for_train_valid


def train(args):
    """
    The train and valid function for the struc2vec
    """
    # Generate the train/valid sample for the training
    samples = generate_samples(args)
    samples = np.array(samples)
    samples = samples[samples[:, 0].argsort()]
    sample_num = len(samples)
    train_sample_num = int(0.8 * sample_num)
    train_data = samples[:train_sample_num]
    valid_data = samples[train_sample_num:]
    train_dataset = MLPDataset(train_data)
    valid_dataset = MLPDataset(valid_data)

    # Construct the model and optimizer
    model = MLPModel(args.w2v_emb_size, args.num_class)
    optimizer = paddle.optimizer.SGD(learning_rate=args.lr,
                                     parameters=model.parameters(),
                                     weight_decay=0.1)

    # Construct the loss and metric 
    crition = CECrition()
    metric = paddle.metric.Accuracy()

    for epoch in range(0, args.epoch):
        # Train the mlp model and calculate the accuracy 
        train_data_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True)
        # Create the dataloader for the mlp task 
        valid_data_loader = DataLoader(
            valid_dataset, batch_size=16, shuffle=False)

        metric.reset()
        for i, (input_embedding, label) in enumerate(train_data_loader):
            logits = model(input_embedding)
            loss = crition(logits, label.unsqueeze(axis=1))
            loss.backward()
            optimizer.step()
            optimizer.clear_grad()
            correct = metric.compute(logits, label)
            metric.update(correct)
        log.info("train process, the epoch:{}, the accuracy:{}".format(
            epoch, metric.accumulate()))

        # Valid the mlp model and calucate the accuracy
        with paddle.no_grad():
            metric.reset()
            for i, (input_embedding, label) in enumerate(valid_data_loader):
                logits = model(input_embedding)
                loss = crition(logits, label.unsqueeze(axis=1))
                correct = metric.compute(logits, label)
                metric.update(correct)
            log.info("valid process, the epoch:{}, the accuracy:{}".format(
                epoch, metric.accumulate()))


def main(args):
    """
    The main fucntion to run the algorithm struc2vec
    """
    if args.generate_emb:
        # Construct the graph with the input file 
        log.info("generate the random walk path and node embedding.")
        dataset = EdgeDataset(
            undirected=args.undirected, data_dir=args.edge_file)
        graph = StrucVecGraph(dataset.graph, dataset.nodes, args.tag,
                              args.opt1, args.opt2, args.opt3, args.depth,
                              args.num_walks, args.walk_depth)

        # Calclute the node similarity by the struc2vec algorithm  
        graph.output_degree_with_depth(args.depth, args.opt1)
        graph.calc_distances_between_nodes()
        graph.normlization_layer_weight()

        # Random walk by the node similarity 
        graph.random_walk_structual_sim()
        learning_embedding_from_struc2vec(args)
        file_label = open(args.label_file)
        file_label_reindex = open(args.label_file + "_reindex", "w")
        for line in file_label:
            items = line.strip("\n\r").split(" ")
            try:
                items = [int(item) for item in items]
            except:
                continue
            if items[0] not in dataset.node_dict:
                continue
            reindex = dataset.node_dict[items[0]]
            file_label_reindex.write(str(reindex) + " " + str(items[1]) + "\n")
        file_label_reindex.close()

    if args.do_train:
        train(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='struc2vec')
    parser.add_argument(
        "--edge_file",
        type=str,
        default="",
        required=True,
        help="The edge file for the graph.")
    parser.add_argument(
        "--label_file",
        type=str,
        default="",
        required=True,
        help="The label file for the label of graph node.")
    parser.add_argument(
        "--emb_file",
        type=str,
        default="w2v_emb",
        help="The file name for the word2vec result.")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="The batch_size for training.")
    parser.add_argument(
        "--undirected",
        type=bool,
        default=True,
        help="Is a undirected graph if set True.")
    parser.add_argument(
        "--depth",
        type=int,
        default=8,
        help="The layer len for the struc2vec.")
    parser.add_argument(
        "--num_walks",
        type=int,
        default=10,
        help="The walk len for the random walk.")
    parser.add_argument(
        "--walk_depth",
        type=int,
        default=80,
        help="The walk depth for the random walk.")
    parser.add_argument(
        "--opt1",
        type=bool,
        default=False,
        help="The optimize flag for reducing time cost.")
    parser.add_argument(
        "--opt2",
        type=bool,
        default=False,
        help="The optimize flag for reducing time cost.")
    parser.add_argument(
        "--opt3",
        type=bool,
        default=False,
        help="The optimize flag for reducing time cost.")
    parser.add_argument(
        "--w2v_emb_size",
        type=int,
        default=128,
        help="The embedding size for the word2vec.")
    parser.add_argument(
        "--w2v_window_size",
        type=int,
        default=10,
        help="The window size for the word2vec.")
    parser.add_argument(
        "--w2v_epoch",
        type=int,
        default=5,
        help="The training epoch for the word2vec.")
    parser.add_argument(
        "--generate_emb",
        type=bool,
        default=False,
        help="The flag for generating the embedding.")
    parser.add_argument(
        "--do_train",
        type=bool,
        default=False,
        help="The flag for traing the classifier with embedding.")
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-1,
        help="The learning_rate for the classifier.")
    parser.add_argument(
        "--num_class",
        type=int,
        default=4,
        help="The number of train data label.")
    parser.add_argument(
        "--epoch", type=int, default=20, help="The epoch for the classifier.")
    parser.add_argument(
        "--tag",
        type=str,
        default="struc2vec",
        help="The prefix name for the embedding file.")

    args = parser.parse_args()
    main(args)
