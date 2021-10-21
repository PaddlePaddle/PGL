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
import logging
import warnings
from collections import defaultdict

import paddle
import numpy as np
import paddle.distributed as dist
from tqdm import tqdm
from visualdl import LogWriter

from dataset.reader import read_trigraph
from dataset.dataset import create_dataloaders
from models.ke_model import KGEModel
from models.loss_func import LossFunction
from config import KGEArgParser
from utils import timer_wrapper, set_seed
from utils import set_logger, print_log, adjust_args


def partial_evaluate(scores, corr_idxs, filter_list):
    logs = []
    for i in range(scores.shape[0]):
        rank = (scores[i] > scores[i][corr_idxs[i]]).astype('float32')
        if filter_list is not None:
            mask = paddle.ones(rank.shape, dtype='float32')
            mask[filter_list[i]] = 0.
            rank = rank * mask
        rank = paddle.sum(rank) + 1
        logs.append({
            'MRR': 1.0 / rank,
            'MR': float(rank),
            'HITS@1': 1.0 if rank <= 1 else 0.0,
            'HITS@3': 1.0 if rank <= 3 else 0.0,
            'HITS@10': 1.0 if rank <= 10 else 0.0,
        })
    return logs


@timer_wrapper('evaluation')
def evaluate(model,
             loader,
             evaluate_mode='test',
             filter_dict=None,
             save_path='./tmp/'):
    """evaluate
    """
    model.eval()
    with paddle.no_grad():
        h_output = defaultdict(list)
        t_output = defaultdict(list)
        h_metrics = []
        t_metrics = []
        dataset_mode = 'normal'
        output = {'h,r->t': {}, 't,r->h': {}, 'average': {}}

        for mode, (h, r, t, cand), corr_idx in tqdm(loader):
            dataset_mode = mode
            if mode == 'wiki':
                score = model.predict(h, r, cand)
                rank = paddle.argsort(score, axis=1, descending=True)
                t_output['t_pred_top10'].append(rank[:, :10].cpu())
                t_output['t_correct_index'].append(corr_idx.cpu())
            else:
                t_score = model.predict(h, r, cand, mode='tail')
                h_score = model.predict(t, r, cand, mode='head')

                if filter_dict is not None:
                    h_filter_list = [
                        filter_dict['head'][(ti, ri)]
                        for ti, ri in zip(t.numpy(), r.numpy())
                    ]
                    t_filter_list = [
                        filter_dict['tail'][(hi, ri)]
                        for hi, ri in zip(h.numpy(), r.numpy())
                    ]
                else:
                    h_filter_list = None
                    t_filter_list = None

                h_metrics += partial_evaluate(h_score, h, h_filter_list)
                t_metrics += partial_evaluate(t_score, t, t_filter_list)

        if dataset_mode == 'wiki':
            for key, value in h_output.items():
                output['t,r->h'][key] = np.concatenate(value, axis=0)
            for key, value in t_output.items():
                output['h,r->t'][key] = np.concatenate(value, axis=0)

            paddle.save(output, save_path)
        else:
            for metric in h_metrics[0].keys():
                output['t,r->h'][metric] = np.mean(
                    [x[metric] for x in h_metrics])
                output['h,r->t'][metric] = np.mean(
                    [x[metric] for x in t_metrics])
                output['average'][metric] = (
                    output['t,r->h'][metric] + output['h,r->t'][metric]) / 2
            print('-------------- %s result --------------' % evaluate_mode)
            print('t,r->h  |', ' '.join(
                ['{}: {}'.format(k, v) for k, v in output['t,r->h'].items()]))
            print('h,r->t  |', ' '.join(
                ['{}: {}'.format(k, v) for k, v in output['h,r->t'].items()]))
            print('average |', ' '.join(
                ['{}: {}'.format(k, v) for k, v in output['average'].items()]))
            print('-----------------------------------------')


def main(writer):
    # def main():
    """Main function for knowledge representation learning
    """
    args = KGEArgParser().parse_args()
    args = adjust_args(args)
    set_seed(args.seed)
    set_logger(args)
    for arg in vars(args):
        logging.info('{:20}:{}'.format(arg, getattr(args, arg)))

    trigraph = read_trigraph(args.data_path, args.data_name)

    if args.filter_sample or args.filter_eval:
        filter_dict = {
            'head': trigraph.true_heads_for_tail_rel,
            'tail': trigraph.true_tails_for_head_rel
        }
    else:
        filter_dict = None

    if dist.get_world_size() > 1:
        dist.init_parallel_env()

    model = KGEModel(args.model_name, trigraph, args)

    # The DataParallel will influence start_async_update or not
    if args.async_update:
        if not args.mix_cpu_gpu:
            raise ValueError(
                "We only support async_update in mix_cpu_gpu mode.")
        print('=' * 20 + '\n Async Update!\n' + '=' * 20)
        model.start_async_update()

    if dist.get_world_size() > 1 and len(model.parameters()) > 0:
        model = paddle.DataParallel(model)
        model = model._layer

    if len(model.parameters()) > 0:
        # There will be some difference of lr / optimizer  on Relation Embedding
        optimizer = paddle.optimizer.Adagrad(
            learning_rate=args.mlp_lr,
            epsilon=1e-10,
            parameters=model.parameters())
    else:
        warnings.warn('there is no model parameter on gpu, optimizer is None.',
                      RuntimeWarning)
        optimizer = None

    loss_func = LossFunction(
        name=args.loss_type,
        pairwise=args.pairwise,
        margin=args.margin,
        neg_adv_spl=args.neg_adversarial_sampling,
        neg_adv_temp=args.adversarial_temperature)

    train_loader, valid_loader, test_loader = create_dataloaders(
        trigraph,
        args,
        filter_dict=filter_dict if args.filter_sample else None,
        shared_ent_path=model.shared_ent_path if args.mix_cpu_gpu else None)

    timer = defaultdict(int)
    log = defaultdict(int)
    ts = t_step = time.time()
    step = 1
    for epoch in range(args.num_epoch):
        model.train()
        for indexes, prefetch_embeddings, mode in train_loader:
            h, r, t, neg_ents, all_ents = indexes
            all_ents_emb, rel_emb = prefetch_embeddings

            if rel_emb is not None:
                rel_emb.stop_gradient = False
            if all_ents_emb is not None:
                all_ents_emb.stop_gradient = False

            timer['sample'] += (time.time() - ts)

            ts = time.time()
            h_emb, r_emb, t_emb, neg_emb, mask = model.prepare_inputs(
                h, r, t, all_ents, neg_ents, all_ents_emb, rel_emb, mode, args)
            pos_score = model.forward(h_emb, r_emb, t_emb)

            writer.add_scalar(
                tag="pos_score", step=step, value=pos_score.sum().numpy()[0])

            if mode == 'head':
                neg_score = model.get_neg_score(t_emb, r_emb, neg_emb, True,
                                                mask)
            else:
                neg_score = model.get_neg_score(h_emb, r_emb, neg_emb, False,
                                                mask)
            neg_score = neg_score.reshape((-1, args.neg_sample_size))

            writer.add_scalar(
                tag="neg_score", step=step, value=neg_score.sum().numpy()[0])

            loss = loss_func(pos_score, neg_score)
            log['loss'] += loss.numpy()[0]

            writer.add_scalar(tag="loss", step=step, value=loss.numpy()[0])

            if args.reg_coef > 0. and args.reg_norm >= 0:
                if all_ents_emb is None:
                    if args.mix_cpu_gpu:
                        ent_params = model.ent_embedding.curr_emb
                    else:
                        ent_params = paddle.concat([
                            h_emb, t_emb, neg_emb.reshape((-1,
                                                           h_emb.shape[-1]))
                        ])
                else:
                    ent_params = all_ents_emb

                if rel_emb is None:
                    rel_params = r_emb
                else:
                    rel_params = rel_emb
                # reg = paddle.norm(params, p=args.reg_norm).pow(args.reg_norm) # 37 steps / s -> partial embedding 102 steps / s -> model return params 98 steps / s
                # reg = params.norm(p=args.reg_norm)**args.reg_norm # 34 steps / s
                # reg = paddle.sum(params.abs().pow(args.reg_norm))
                reg = paddle.sum(ent_params.abs().pow(args.reg_norm)) + \
                    paddle.sum(rel_params.abs().pow(args.reg_norm))
                reg = args.reg_coef * reg
                log['reg'] += reg.numpy()[0]

                writer.add_scalar(tag="reg", step=step, value=reg.numpy()[0])

                loss = loss + reg
            timer['forward'] += (time.time() - ts)

            ts = time.time()
            loss.backward()
            timer['backward'] += (time.time() - ts)

            ts = time.time()
            if optimizer is not None:
                optimizer.step()

            ent_trace = (all_ents.numpy(), all_ents_emb.grad.numpy()) \
                if all_ents_emb is not None else None
            rel_trace = (r.numpy(), rel_emb.grad.numpy()) \
                if rel_emb is not None else None

            model.step(ent_trace, rel_trace)

            if optimizer is not None:
                optimizer.clear_grad()

            timer['update'] += (time.time() - ts)

            if (step + 1) % args.log_interval == 0:
                print_log(step, args.log_interval, log, timer, t_step)
                timer = defaultdict(int)
                log = defaultdict(int)
                t_step = time.time()

            if args.valid and (step + 1) % args.eval_interval == 0:
                evaluate(model, valid_loader, 'valid', filter_dict
                         if args.filter_eval else None)

            step += 1
            ts = time.time()

    if args.async_update:
        model.finish_async_update()

    if args.test:
        evaluate(model, test_loader, 'test', filter_dict
                 if args.filter_eval else None,
                 os.path.join(args.save_path, 'test.pkl'))


if __name__ == '__main__':
    with LogWriter(logdir="./log/rotate_sgpu/train") as writer:
        main(writer)
    # main()
