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
import argparse
import time
import math
import os
import io
from multiprocessing import Pool
import glob

import numpy as np
import sklearn.metrics
from sklearn.metrics import f1_score

import pgl
from pgl import data_loader
from pgl.utils import op
from pgl.utils.logger import log
import paddle.fluid as fluid
import paddle.fluid.layers as l


def load(name):
    if name == "BlogCatalog":
        dataset = data_loader.BlogCatalogDataset()
    elif name == "ArXiv":
        dataset = data_loader.ArXivDataset()
    else:
        raise ValueError(name + " dataset doesn't exists")
    return dataset


def deepwalk_model(graph, hidden_size=16, neg_num=5):

    pyreader = l.py_reader(
        capacity=70,
        shapes=[[-1, 1, 1], [-1, 1, 1], [-1, neg_num, 1]],
        dtypes=['int64', 'int64', 'int64'],
        lod_levels=[0, 0, 0],
        name='train',
        use_double_buffer=True)

    embed_init = fluid.initializer.UniformInitializer(low=-1.0, high=1.0)
    weight_init = fluid.initializer.TruncatedNormal(scale=1.0 /
                                                    math.sqrt(hidden_size))

    src, pos, negs = l.read_file(pyreader)

    embed_src = l.embedding(
        input=src,
        size=[graph.num_nodes, hidden_size],
        param_attr=fluid.ParamAttr(
            name='content', initializer=embed_init))

    weight_pos = l.embedding(
        input=pos,
        size=[graph.num_nodes, hidden_size],
        param_attr=fluid.ParamAttr(
            name='weight', initializer=weight_init))
    weight_negs = l.embedding(
        input=negs,
        size=[graph.num_nodes, hidden_size],
        param_attr=fluid.ParamAttr(
            name='weight', initializer=weight_init))

    pos_logits = l.matmul(
        embed_src, weight_pos, transpose_y=True)  # [batch_size, 1, 1]
    neg_logits = l.matmul(
        embed_src, weight_negs, transpose_y=True)  # [batch_size, 1, neg_num]

    ones_label = pos_logits * 0. + 1.
    ones_label.stop_gradient = True
    pos_loss = l.sigmoid_cross_entropy_with_logits(pos_logits, ones_label)

    zeros_label = neg_logits * 0.
    zeros_label.stop_gradient = True
    neg_loss = l.sigmoid_cross_entropy_with_logits(neg_logits, zeros_label)
    loss = (l.reduce_mean(pos_loss) + l.reduce_mean(neg_loss)) / 2

    return pyreader, loss


def gen_pair(walks, left_win_size=2, right_win_size=2):
    src = []
    pos = []
    for walk in walks:
        for left_offset in range(1, left_win_size + 1):
            src.extend(walk[left_offset:])
            pos.extend(walk[:-left_offset])
        for right_offset in range(1, right_win_size + 1):
            src.extend(walk[:-right_offset])
            pos.extend(walk[right_offset:])
    src, pos = np.array(src, dtype=np.int64), np.array(pos, dtype=np.int64)
    src, pos = np.expand_dims(src, -1), np.expand_dims(pos, -1)
    src, pos = np.expand_dims(src, -1), np.expand_dims(pos, -1)
    return src, pos


def deepwalk_generator(graph,
                       batch_size=512,
                       walk_len=5,
                       win_size=2,
                       neg_num=5,
                       epoch=200,
                       filelist=None):
    def walks_generator():
        if filelist is not None:
            bucket = []
            for filename in filelist:
                with io.open(filename) as inf:
                    for line in inf:
                        walk = [int(x) for x in line.strip('\n').split(' ')]
                        bucket.append(walk)
                        if len(bucket) == batch_size:
                            yield bucket
                            bucket = []
            if len(bucket):
                yield bucket
        else:
            for _ in range(epoch):
                for nodes in graph.node_batch_iter(batch_size):
                    walks = graph.random_walk(nodes, walk_len)
                    yield walks

    def wrapper():
        for walks in walks_generator():
            src, pos = gen_pair(walks, win_size, win_size)
            if src.shape[0] == 0:
                continue
            negs = graph.sample_nodes([len(src), neg_num, 1]).astype(np.int64)
            yield [src, pos, negs]

    return wrapper


def process(args):
    idx, graph, save_path, epoch, batch_size, walk_len, seed = args
    with open('%s/%s' % (save_path, idx), 'w') as outf:
        for _ in range(epoch):
            np.random.seed(seed)
            for nodes in graph.node_batch_iter(batch_size):
                walks = graph.random_walk(nodes, walk_len)
                for walk in walks:
                    outf.write(' '.join([str(token) for token in walk]) + '\n')


def main(args):
    hidden_size = args.hidden_size
    neg_num = args.neg_num
    epoch = args.epoch
    save_path = args.save_path
    batch_size = args.batch_size
    walk_len = args.walk_len
    win_size = args.win_size

    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    dataset = load(args.dataset)

    if args.offline_learning:
        log.info("Start random walk on disk...")
        walk_save_path = os.path.join(save_path, "walks")
        if not os.path.isdir(walk_save_path):
            os.makedirs(walk_save_path)
        pool = Pool(args.processes)
        args_list = [(x, dataset.graph, walk_save_path, 1, batch_size,
                      walk_len, np.random.randint(
                          2**32, dtype="int64")) for x in range(epoch)]
        pool.map(process, args_list)
        filelist = glob.glob(os.path.join(walk_save_path, "*"))
        log.info("Random walk on disk Done.")
    else:
        filelist = None

    train_steps = int(dataset.graph.num_nodes / batch_size) * epoch

    place = fluid.CUDAPlace(0) if args.use_cuda else fluid.CPUPlace()
    deepwalk_prog = fluid.Program()
    startup_prog = fluid.Program()

    with fluid.program_guard(deepwalk_prog, startup_prog):
        with fluid.unique_name.guard():
            deepwalk_pyreader, deepwalk_loss = deepwalk_model(
                dataset.graph, hidden_size=hidden_size, neg_num=neg_num)
            lr = l.polynomial_decay(0.025, train_steps, 0.0001)
            adam = fluid.optimizer.Adam(lr)
            adam.minimize(deepwalk_loss)

    deepwalk_pyreader.decorate_tensor_provider(
        deepwalk_generator(
            dataset.graph,
            batch_size=batch_size,
            walk_len=walk_len,
            win_size=win_size,
            epoch=epoch,
            neg_num=neg_num,
            filelist=filelist))

    deepwalk_pyreader.start()

    exe = fluid.Executor(place)
    exe.run(startup_prog)

    prev_time = time.time()
    step = 0

    while 1:
        try:
            deepwalk_loss_val = exe.run(deepwalk_prog,
                                        fetch_list=[deepwalk_loss],
                                        return_numpy=True)[0]
            cur_time = time.time()
            use_time = cur_time - prev_time
            prev_time = cur_time
            step += 1
            log.info("Step %d " % step + "Deepwalk Loss: %f " %
                     deepwalk_loss_val + " %f s/step." % use_time)
        except fluid.core.EOFException:
            deepwalk_pyreader.reset()
            break

    fluid.io.save_persistables(exe,
                               os.path.join(save_path, "paddle_model"),
                               deepwalk_prog)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='deepwalk')
    parser.add_argument(
        "--dataset",
        type=str,
        default="ArXiv",
        help="dataset (BlogCatalog, ArXiv)")
    parser.add_argument("--use_cuda", action='store_true', help="use_cuda")
    parser.add_argument(
        "--offline_learning", action='store_true', help="use_cuda")
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--neg_num", type=int, default=20)
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--walk_len", type=int, default=40)
    parser.add_argument("--win_size", type=int, default=10)
    parser.add_argument("--save_path", type=str, default="./tmp/deepwalk")
    parser.add_argument("--processes", type=int, default=10)
    args = parser.parse_args()
    log.info(args)
    main(args)
