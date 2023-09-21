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
import sys
import time
import warnings
from collections import defaultdict

import paddle
import numpy as np
import paddle.nn as nn
import paddle.distributed as dist
from paddle.optimizer.lr import StepDecay

from dataset.reader import read_trigraph
from dataset.dataset import create_dataloaders
from models.ke_model import KGEModel
from models.loss_func import LossFunction
from utils import set_seed, set_logger, print_log
from utils import evaluate
from config import prepare_config


class KEModelDP(nn.Layer):
    """
    KEModel for DataParallel mode.
    """

    def __init__(self, model, args):
        super(KEModelDP, self).__init__()
        self.model = model
        self.args = args
        self.loss_func = LossFunction(
            name=args.loss_type,
            pairwise=args.pairwise,
            margin=args.margin,
            neg_adv_spl=args.neg_adversarial_sampling,
            neg_adv_temp=args.adversarial_temperature)

    def forward(self, h, r, t, all_ents, neg_ents, all_ents_emb, rel_emb, mode,
                weights):
        h_emb, r_emb, t_emb, neg_emb, mask = self.model.prepare_inputs(
            h, r, t, all_ents, neg_ents, all_ents_emb, rel_emb, mode,
            self.args)
        pos_score = self.model.forward(h_emb, r_emb, t_emb)

        if mode == 'head':
            neg_score = self.model.get_neg_score(t_emb, r_emb, neg_emb, True,
                                                 mask)
        elif mode == 'tail':
            neg_score = self.model.get_neg_score(h_emb, r_emb, neg_emb, False,
                                                 mask)
        else:
            raise ValueError('Unsupported negative mode {}.'.format(mode))
        neg_score = neg_score.reshape([self.args.batch_size, -1])

        loss = self.loss_func(pos_score, neg_score, weights)
        if self.args.use_embedding_regularization:
            reg_loss = self.model.get_regularization(h_emb, r_emb, t_emb,
                                                     neg_emb)

            loss = loss + reg_loss
        return loss


def main():
    """Main function for shallow knowledge embedding methods.
    """
    args = prepare_config()

    if dist.get_world_size() > 1:
        dist.init_parallel_env()

    set_seed(args.seed)
    set_logger(args)

    trigraph = read_trigraph(args.data_path, args.data_name, args.use_dict,
                             args.kv_mode)
    if args.valid_percent < 1:
        trigraph.sampled_subgraph(args.valid_percent, dataset='valid')

    use_filter_set = args.filter_sample or args.filter_eval or args.weighted_loss
    if use_filter_set:
        filter_dict = {
            'head': trigraph.true_heads_for_tail_rel,
            'tail': trigraph.true_tails_for_head_rel
        }
    else:
        filter_dict = None

    model = KGEModel(args.model_name, trigraph, args)

    if args.async_update:
        model.start_async_update()

    if dist.get_world_size() > 1 and len(model.parameters()) > 0:
        dpmodel = paddle.DataParallel(KEModelDP(model, args))
    else:
        dpmodel = KEModelDP(model, args)

    if len(dpmodel.parameters()) > 0:
        if args.optimizer == 'adam':
            optim_func = paddle.optimizer.Adam
        elif args.optimizer == 'adagrad':
            optim_func = paddle.optimizer.Adagrad
        elif args.optimizer == 'sgd':
            optim_func = paddle.optimizer.SGD
        else:
            errors = 'Optimizer {} not supported!'.format(args.optimizer)
            raise ValueError(errors)
        if args.scheduler_interval > 0:
            scheduler = StepDecay(
                learning_rate=args.lr,
                step_size=args.scheduler_interval,
                gamma=0.5,
                last_epoch=-1,
                verbose=True)
            optimizer = optim_func(
                learning_rate=scheduler,
                epsilon=1e-10,
                parameters=dpmodel.parameters())
        else:
            if args.optimizer in {'sgd'}:
                optimizer = optim_func(
                    learning_rate=args.lr, parameters=dpmodel.parameters())
            else:
                optimizer = optim_func(
                    learning_rate=args.lr,
                    epsilon=1e-10,
                    parameters=dpmodel.parameters())
    else:
        warnings.warn('There is no model parameter on gpu, optimizer is None.',
                      RuntimeWarning)
        optimizer = None

    train_loader, valid_loader, test_loader = create_dataloaders(
        trigraph,
        args,
        filter_dict=filter_dict if use_filter_set else None,
        shared_ent_path=model.shared_ent_path if args.mix_cpu_gpu else None)

    timer = defaultdict(int)
    log = defaultdict(int)
    ts = t_step = time.time()
    step = 1
    stop = False
    dev_id = int(paddle.device.get_device().split(":")[1])
    for epoch in range(args.num_epoch):
        for indexes, prefetch_embeddings, mode in train_loader:
            h, r, t, neg_ents, all_ents = indexes
            all_ents_emb, rel_emb, weights = prefetch_embeddings

            r = r.cuda(dev_id)
            if all_ents is not None:
                all_ents = all_ents.cuda(dev_id)

            if rel_emb is not None:
                rel_emb = rel_emb.cuda(dev_id)
                rel_emb.stop_gradient = False
            if all_ents_emb is not None:
                all_ents_emb = all_ents_emb.cuda(dev_id)
                all_ents_emb.stop_gradient = False
            timer['sample'] += (time.time() - ts)

            ts = time.time()
            loss = dpmodel(h, r, t, all_ents, neg_ents, all_ents_emb, rel_emb,
                           mode, weights)
            timer['forward'] += (time.time() - ts)

            log['loss'] += float(loss)

            ts = time.time()
            loss.backward()
            timer['backward'] += (time.time() - ts)

            ts = time.time()
            if optimizer is not None:
                optimizer.step()
                optimizer.clear_grad()

            if args.mix_cpu_gpu:
                ent_trace, rel_trace = model.create_trace(
                    all_ents, all_ents_emb, r, rel_emb)
                model.step(ent_trace, rel_trace)
            else:
                model.step()

            timer['update'] += (time.time() - ts)

            if args.log_interval > 0 and (step + 1) % args.log_interval == 0:
                print_log(step, args.log_interval, log, timer,
                          time.time() - t_step)
                timer = defaultdict(int)
                log = defaultdict(int)
                t_step = time.time()

            if args.valid and dist.get_rank() == 0 and (
                    step + 1) % args.eval_interval == 0:
                evaluate(
                    model,
                    valid_loader,
                    'valid',
                    filter_dict if args.filter_eval else None,
                    data_mode=args.data_name)

            if args.scheduler_interval > 0 and step % args.scheduler_interval == 0:
                scheduler.step()

            step += 1
            if dist.get_rank() == 0:
                if args.save_interval > 0 and step % args.save_interval == 0:
                    model.save(args.step_path)
            if step >= args.max_steps:
                stop = True
                break

            ts = time.time()
        if stop:
            break

    if args.async_update:
        model.finish_async_update()

    if args.test and dist.get_rank() == 0:
        evaluate(
            model,
            test_loader,
            'test',
            filter_dict if args.filter_eval else None,
            os.path.join(args.save_path, 'test.pkl'),
            data_mode=args.data_name)


if __name__ == '__main__':
    main()
