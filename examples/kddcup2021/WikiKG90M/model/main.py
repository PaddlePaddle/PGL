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
import logging
import time
import pdb

from models.base_model import BaseKEModel
from dataloader import TrainDataset, EvalDataset, NewBidirectionalOneShotIterator
from dataloader import get_dataset
from optimizer import NumpySgdOptimizer, NumpyAdagradOptimizer
from utils import get_compatible_batch_size, CommonArgParser

import paddle
import random
from math import ceil
from tqdm import tqdm
import numpy as np
from collections import defaultdict
from ogb.lsc import WikiKG90MDataset, WikiKG90MEvaluator


def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)


def prepare_save_path(args):
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    folder = '{}_{}_d_{}_g_{}'.format(args.model_name, args.dataset,
                                      args.hidden_dim, args.gamma)
    n = len([x for x in os.listdir(args.save_path) if x.startswith(folder)])
    folder += str(n)
    args.save_path = os.path.join(args.save_path, folder)

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)


def set_logger(args):
    '''
    Write logs to console and log file
    '''
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
    args = ArgParser().parse_args()
    prepare_save_path(args)
    args.neg_sample_size_eval = 1000
    set_global_seed(args.seed)

    init_time_start = time.time()
    dataset = get_dataset(args, args.data_path, args.dataset, args.format,
                          args.delimiter, args.data_files,
                          args.has_edge_importance)
    args.batch_size = get_compatible_batch_size(args.batch_size,
                                                args.neg_sample_size)
    args.batch_size_eval = get_compatible_batch_size(args.batch_size_eval,
                                                     args.neg_sample_size_eval)

    #print(args)
    set_logger(args)

    print("To build training dataset")
    t1 = time.time()
    train_data = TrainDataset(
        dataset, args, has_importance=args.has_edge_importance)
    print("Training dataset built, it takes %s seconds" % (time.time() - t1))
    args.num_workers = 8  # fix num_worker to 8
    print("Building training sampler")
    t1 = time.time()
    train_sampler_head = train_data.create_sampler(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        neg_sample_size=args.neg_sample_size,
        neg_mode='head')
    train_sampler_tail = train_data.create_sampler(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        neg_sample_size=args.neg_sample_size,
        neg_mode='tail')
    train_sampler = NewBidirectionalOneShotIterator(train_sampler_head,
                                                    train_sampler_tail)
    print("Training sampler created, it takes %s seconds" % (time.time() - t1))

    if args.valid or args.test:
        if len(args.gpu) > 1:
            args.num_test_proc = args.num_proc if args.num_proc < len(
                args.gpu) else len(args.gpu)
        else:
            args.num_test_proc = args.num_proc
        print("To create eval_dataset")
        t1 = time.time()
        eval_dataset = EvalDataset(dataset, args)
        print("eval_dataset created, it takes %d seconds" % (time.time() - t1))

    if args.valid:
        if args.num_proc > 1:
            valid_samplers = []
            for i in range(args.num_proc):
                print("creating valid sampler for proc %d" % i)
                t1 = time.time()
                valid_sampler_tail = eval_dataset.create_sampler(
                    'valid',
                    args.batch_size_eval,
                    mode='tail',
                    num_workers=args.num_workers,
                    rank=i,
                    ranks=args.num_proc)
                valid_samplers.append(valid_sampler_tail)
                print("Valid sampler for proc %d created, it takes %s seconds"
                      % (i, time.time() - t1))
        else:
            valid_sampler_tail = eval_dataset.create_sampler(
                'valid',
                args.batch_size_eval,
                mode='tail',
                num_workers=args.num_workers,
                rank=0,
                ranks=1)
            valid_samplers = [valid_sampler_tail]

    for arg in vars(args):
        logging.info('{:20}:{}'.format(arg, getattr(args, arg)))

    print("To create model")
    t1 = time.time()
    model = BaseKEModel(
        args=args,
        n_entities=dataset.n_entities,
        n_relations=dataset.n_relations,
        model_name=args.model_name,
        hidden_size=args.hidden_dim,
        entity_feat_dim=dataset.entity_feat.shape[1],
        relation_feat_dim=dataset.relation_feat.shape[1],
        gamma=args.gamma,
        double_entity_emb=args.double_ent,
        relation_times=args.ote_size,
        scale_type=args.scale_type)

    model.entity_feat = dataset.entity_feat
    model.relation_feat = dataset.relation_feat
    print(len(model.parameters()))

    if args.cpu_emb:
        print("using cpu emb\n" * 5)
    else:
        print("using gpu emb\n" * 5)
    optimizer = paddle.optimizer.Adam(
        learning_rate=args.mlp_lr, parameters=model.parameters())
    lr_tensor = paddle.to_tensor(args.lr)

    global_step = 0
    tic_train = time.time()
    log = {}
    log["loss"] = 0.0
    log["regularization"] = 0.0
    for step in range(0, args.max_step):
        pos_triples, neg_triples, ids, neg_head = next(train_sampler)
        loss = model.forward(pos_triples, neg_triples, ids, neg_head)

        log["loss"] = loss.numpy()[0]
        if args.regularization_coef > 0.0 and args.regularization_norm > 0:
            coef, nm = args.regularization_coef, args.regularization_norm
            reg = coef * norm(model.entity_embedding.curr_emb, nm)
            log['regularization'] = reg.numpy()[0]
            loss = loss + reg

        loss.backward()
        optimizer.step()
        if args.cpu_emb:
            model.entity_embedding.step(lr_tensor)
        optimizer.clear_grad()
        if (step + 1) % args.log_interval == 0:
            speed = args.log_interval / (time.time() - tic_train)
            logging.info(
                "step: %d, train loss: %.5f, regularization: %.4e, speed: %.2f steps/s"
                % (step, log["loss"], log["regularization"], speed))
            log["loss"] = 0.0
            tic_train = time.time()

        if args.valid and (
                step + 1
        ) % args.eval_interval == 0 and step > 1 and valid_samplers is not None:
            print("Valid begin")
            valid_start = time.time()
            valid_input_dict = test(
                args, model, valid_samplers, step, rank=0, mode='Valid')
            paddle.save(valid_input_dict,
                        os.path.join(args.save_path,
                                     "valid_{}.pkl".format(step)))
            # Save the model for the inference
        if (step + 1) % args.save_step == 0:
            print("The step:{}, save model path:{}".format(step + 1,
                                                           args.save_path))
            model.save_model()
            print("Save model done.")


def test(args, model, test_samplers, step, rank=0, mode='Test'):
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
                log, score = model.forward_test_wikikg(query, ans, candidate,
                                                       sampler.mode)
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
