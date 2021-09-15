# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
import time
import random
import logging
from math import ceil
from collections import defaultdict

import paddle
import numpy as np
from tqdm import tqdm
import paddle.distributed as dist
from ogb.lsc import WikiKG90MDataset, WikiKG90MEvaluator
from paddle.io import DataLoader

from models.base_model import BaseKEModel, NumpyEmbedding
from utils.helper import get_compatible_batch_size, CommonArgParser
from utils.helper import prepare_save_path, get_save_path
from dataset import load_dataset
from utils.dataset import KGDataset, TestKGDataset


def set_global_seed(seed):
    """seed"""
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)


def set_logger(args):
    """
    Write logs to console and log file
    """
    log_file = os.path.join(args.save_path, 'train.log')
    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_file,
        filemode='a+')
    if args.print_on_screen:
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter(
            '%(asctime)s %(levelname)-8s %(message)s')
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)


class ArgParser(CommonArgParser):
    """parser"""
    def __init__(self):
        super(ArgParser, self).__init__()

        self.add_argument(
            '--gpu',
            type=int,
            default=[-1],
            nargs='+',
            help='A list of gpu ids, e.g. 0 1 2 4')
        self.add_argument('--mix_cpu_gpu', action='store_true',
                          help='Training a knowledge graph embedding model with both CPUs and GPUs.'\
                                  'The embeddings are stored in CPU memory and the training is performed in GPUs.'\
                                  'This is usually used for training a large knowledge graph embeddings.')
        self.add_argument(
            '--valid',
            action='store_true',
            help='Evaluate the model on the validation set in the training.')
        self.add_argument(
            '--rel_part',
            action='store_true',
            help='Enable relation partitioning for multi-GPU training.')
        self.add_argument('--async_update', action='store_true',
                          help='Allow asynchronous update on node embedding for multi-GPU training.'\
                                  'This overlaps CPU and GPU computation to speed up.')
        self.add_argument('--has_edge_importance', action='store_true',
                          help='Allow providing edge importance score for each edge during training.'\
                                  'The positive score will be adjusted '\
                                  'as pos_score = pos_score * edge_importance')

        self.add_argument('--print_on_screen', action='store_true')
        self.add_argument(
            '--mlp_lr',
            type=float,
            default=0.0001,
            help='The learning rate of optimizing mlp')
        self.add_argument('--seed', type=int, default=0, help='random seed')


norm = lambda x, p: x.norm(p=p)**p


def main():
    """main
    """
    args = ArgParser().parse_args()
    # initialize the parallel environment
    if dist.get_world_size() > 1:
        dist.init_parallel_env()
    args.save_path = get_save_path(args)

    args.neg_sample_size_eval = 1000
    set_global_seed(args.seed)

    init_time_start = time.time()
    dataset = load_dataset(args.data_path, args.dataset)
    args.batch_size = get_compatible_batch_size(args.batch_size,
                                                args.neg_sample_size)
    args.batch_size_eval = get_compatible_batch_size(args.batch_size_eval,
                                                     args.neg_sample_size_eval)

    args.weight_path = os.path.join(args.save_path, '__cpu_embedding.npy')
    if dist.get_rank() == 0:
        prepare_save_path(args)
        init_value = (args.gamma + 2.0) / args.hidden_dim
        NumpyEmbedding.create_emb(dataset.graph.num_ents, args.hidden_dim,
                                  init_value, args.scale_type,
                                  args.weight_path)
    else:
        while True:
            if os.path.exists(args.weight_path):
                break
            time.sleep(5)

    #print(args)
    set_logger(args)

    print("To build training dataset")
    t1 = time.time()
    # train_data = TrainDataset(
    #     dataset, args, has_importance=args.has_edge_importance)
    train_data = KGDataset(dataset.graph.train,
                           dataset.graph.num_ents,
                           args.neg_sample_size,
                           neg_mode='full',
                           filter_mode=False)
    print("Training dataset built, it takes %s seconds" % (time.time() - t1))
    args.num_workers = 0  # fix num_worker to 8

    print("Building training sampler")
    t1 = time.time()
    train_sampler = DataLoader(train_data, 
                               batch_size=args.batch_size,
                               shuffle=True,
                               drop_last=True,
                               num_workers=args.num_workers,
                               collate_fn=train_data.mixed_collate_fn)
    print("Training sampler created, it takes %s seconds" % (time.time() - t1))

    if args.valid or args.test:
        if len(args.gpu) > 1:
            args.num_test_proc = args.num_proc if args.num_proc < len(
                args.gpu) else len(args.gpu)
        else:
            args.num_test_proc = args.num_proc
        print("To create eval_dataset")
        t1 = time.time()
        eval_data = TestKGDataset(dataset.graph.valid, dataset.graph.num_ents)
        test_data = TestKGDataset(dataset.graph.test, dataset.graph.num_ents)
        print("eval_dataset created, it takes %d seconds" % (time.time() - t1))

    if args.valid:
        valid_sampler = DataLoader(eval_data, 
                                   batch_size=args.batch_size_eval,
                                   shuffle=True,
                                   drop_last=True,
                                   num_workers=args.num_workers,
                                   collate_fn=eval_data.collate_fn)
        print("Valid sampler created, it takes %f seconds" % (time.time() - t1))

    if args.test:
        test_sampler = DataLoader(test_data, 
                                  batch_size=args.batch_size_eval,
                                  shuffle=True,
                                  drop_last=True,
                                  num_workers=args.num_workers,
                                  collate_fn=test_data.collate_fn)
        print("Test sampler created, it takes %f seconds" % (time.time() - t1))

    for arg in vars(args):
        logging.info('{:20}:{}'.format(arg, getattr(args, arg)))

    print("To create model")
    t1 = time.time()
    model = BaseKEModel(
        args=args,
        n_entities=dataset.graph.num_ents,
        n_relations=dataset.graph.num_rels,
        model_name=args.model_name,
        hidden_size=args.hidden_dim,
        entity_feat_dim=0,
        relation_feat_dim=0,
        gamma=args.gamma,
        double_entity_emb=args.double_ent,
        relation_times=args.ote_size,
        scale_type=args.scale_type)

    model.entity_feat = dataset.graph.ent_feat
    model.relation_feat = dataset.graph.rel_feat
    print(len(model.parameters()))

    if dist.get_world_size() > 1:
        model = paddle.DataParallel(model)

    if args.cpu_emb:
        print("using cpu emb\n" * 5)
    else:
        print("using gpu emb\n" * 5)
    optimizer = paddle.optimizer.Adam(
        learning_rate=args.mlp_lr, parameters=model.parameters())

    global_step = 0
    log = {}
    log["loss"] = 0.0
    log["regularization"] = 0.0
    t_sample = 0
    t_forward = 0
    t_backward = 0
    t_update = 0
    tic_train = time.time()
    for step in range(0, args.max_step):
        for ents, ids, neg_head in train_sampler:
            ts = time.time()
            t_sample += (time.time() - ts)

            pos_triples = ents[:3]
            neg_ents = ents[3]

            ts = time.time()
            loss = model.forward(pos_triples, neg_ents, ids, neg_head)

            log["loss"] = loss.numpy()[0]
            if args.regularization_coef > 0.0 and args.regularization_norm > 0:
                coef, nm = args.regularization_coef, args.regularization_norm
                if type(model) is paddle.DataParallel:
                    curr_emb = model._layers.entity_embedding.curr_emb
                else:
                    curr_emb = model.entity_embedding.curr_emb
                reg = coef * norm(curr_emb, nm)
                log['regularization'] = reg.numpy()[0]
                loss = loss + reg
            t_forward += (time.time() - ts)

            ts = time.time()
            loss.backward()
            t_backward += (time.time() - ts)

            ts = time.time()
            optimizer.step()
            if args.cpu_emb:
                if type(model) is paddle.DataParallel:
                    model._layers.entity_embedding.step(args.lr)
                else:
                    model.entity_embedding.step(args.lr)
            t_update += (time.time() - ts)
            optimizer.clear_grad()

        if (step + 1) % args.log_interval == 0:
            alltime = (time.time() - tic_train)
            speed = args.log_interval / alltime
            logging.info(
                "step: %d, train loss: %.5f, regularization: %.4e, speed: %.2f steps/s"
                % (step, log["loss"], log["regularization"], speed))
            logging.info(
                "step: %d, sample: %f, forward: %f, backward: %f, update: %f" %
                (step, t_sample, t_forward, t_backward, t_update))
            logging.info("%d steps take %f seconds" %
                         (args.log_interval, time.time() - tic_train))
            log["loss"] = 0.0
            t_sample = 0
            t_forward = 0
            t_backward = 0
            t_update = 0
            tic_train = time.time()

        if args.valid and (
                step + 1
        ) % args.eval_interval == 0 and step > 1 and valid_sampler is not None:
            print("Valid begin")
            valid_start = time.time()
            valid_input_dict = test(
                args, model, [valid_sampler], step, rank=0, mode='Valid')
            paddle.save(valid_input_dict,
                        os.path.join(args.save_path,
                                     "valid_{}.pkl".format(step)))

            # Save the model for the inference
        if (step + 1) % args.save_step == 0:
            print("The step:{}, save model path:{}".format(step + 1,
                                                           args.save_path))
            model.save_model()
            print("Save model done.")

    if args.test and test_sampler is not None:
        print("Test begin")
        test_start = time.time()
        test_input_dict = test(
            args,
            model, [test_sampler],
            args.max_step,
            rank=0,
            mode='Test')
        paddle.save(test_input_dict, os.path.join(args.save_path, "test.pkl"))


def test(args, model, test_samplers, step, rank=0, mode='Test'):
    """test
    """
    with paddle.no_grad():
        logs = defaultdict(list)
        answers = defaultdict(list)
        scores = defaultdict(list)
        for sampler in test_samplers:
            print(sampler.num_edges, sampler.batch_size)
            for query, ans, candidate in tqdm(
                    sampler,
                    disable=not args.print_on_screen,
                    total=ceil(sampler.num_edges / sampler.batch_size)):
                if type(model) is paddle.DataParallel:
                    log, score = model._layers.forward_test_wikikg(
                        query, ans, candidate, sampler.mode)
                else:
                    log, score = model.forward_test_wikikg(
                        query, ans, candidate, sampler.mode)
                log = log.cpu()
                score = score.cpu()
                logs[sampler.mode].append(log)
                answers[sampler.mode].append(ans)
                scores[sampler.mode].append(score)
        print("[{}] finished {} forward".format(rank, mode))

        input_dict = {}
        assert len(answers) == 1
        assert 'h,r->t' in answers
        if 'h,r->t' in answers:
            assert 'h,r->t' in logs, "h,r->t not in logs"
            input_dict['h,r->t'] = {
                't_correct_index': paddle.concat(answers['h,r->t'], 0),
                't_pred_top10': paddle.concat(logs['h,r->t'], 0)
            }
            if step >= 30000:
                input_dict['h,r->t']['scores'] = paddle.concat(
                    scores["h,r->t"], 0)

    for i in range(len(test_samplers)):
        test_samplers[i] = test_samplers[i].reset()

    return input_dict


if __name__ == "__main__":
    main()
