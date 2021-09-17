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
from collections import defaultdict

import paddle
import numpy as np
import paddle.distributed as dist
from paddle.io import DataLoader
from tqdm import tqdm

from dataset import load_dataset
from utils.dataset import KGDataset, TestKGDataset
from models.base_model import NumpyEmbedding, KGEModel
from models.base_loss import LossFunction
from utils.helper import CommonArgParser, prepare_save_path, timer_wrapper


def set_seed(seed):
    """Set seed for reproduction

    Execute :code:`export FLAGS_cudnn_deterministic=True` before training command.

    """
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def set_logger(args):
    """Write logs to console and log file
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


def adjust_args(args):
    """Adjust arguments for compatiblity
    """
    args.save_path = prepare_save_path(args)
    args.embs_path = os.path.join(args.save_path, '__cpu_embedding.npy')
    return args


@timer_wrapper('NumPy embedding initialization')
def prepare_cpu_embeddings(args, data):
    """Create NumPy Embedding with main process
    """
    if dist.get_rank() == 0:
        init_value = (args.gamma + 2.0) / args.embed_dim
        NumpyEmbedding.create_emb(
            data.num_ents,
            args.embed_dim,
            init_value,
            weight_path=args.embs_path,
            scale_type=args.scale_type)
    else:
        while True:
            if os.path.exists(args.embs_path):
                break
            time.sleep(5)


def print_log(step, interval, log, timer, t_step):
    """Print log
    """
    time_sum = time.time() - t_step
    logging.info('step: %d, loss: %.5f, reg: %.4e, speed: %.2f steps/s' %
                 (step, log['loss'], log['reg'], interval / time_sum))
    logging.info('timer | sample: %f, forward: %f, backward: %f, update: %f' %
                 (timer['sample'], timer['forward'], timer['backward'],
                  timer['update']))


@timer_wrapper('evaluation')
def test(model, loader, pos_dict=None, save_path='./tmp/'):
    """test
    """
    with paddle.no_grad():
        h_output = defaultdict(list)
        t_output = defaultdict(list)
        output = {'h,r->t': {}, 't,r->h': {}}

        if isinstance(model, paddle.DataParallel):
            model = model._layers

        for mode, (h, r, t, cand), corr_idx in tqdm(loader):
            if mode == 'wiki':
                score = model.predict(h, r, cand)
                rank = paddle.argsort(score, axis=1, descending=True)
                t_output['t_pred_top10'].append(rank.cpu().numpy())
                t_output['t_correct_index'].append(corr_idx.cpu().numpy())
            else:
                t_score = model.predict(h, r, cand, mode='tail')
                h_score = model.predict(t, r, cand, mode='head')
                if pos_dict is not None:
                    t_mask = np.ones(t_rank.shape)
                    h_mask = np.ones(h_rank.shape)
                    h_dict = pos_dict['head']
                    t_dict = pos_dict['tail']
                    for i, (hi, ri, ti,
                            ci) in enumerate(zip(h, r, t, corr_idx)):
                        t_mask[i][t_dict[(hi, ri)]] = 0.
                        h_mask[i][h_dict[(ti, ri)]] = 0.
                        t_mask[i][ci] = 1.
                        h_mask[i][ci] = 1.
                    t_score = t_score * t_mask
                    h_score = h_score * h_mask
                t_rank = paddle.argsort(t_score, axis=1, descending=True)
                h_rank = paddle.argsort(h_score, axis=1, descending=True)
                t_output['rank'].append(t_rank.cpu().numpy())
                h_output['rank'].append(h_rank.cpu().numpy())
                t_output['corr'].append(t.cpu().numpy())
                h_output['corr'].append(h.cpu().numpy())

        for key, value in h_output.items():
            output['t,r->h'][key] = np.concatenate(value, axis=0)
        for key, value in t_output.items():
            output['h,r->t'][key] = np.concatenate(value, axis=0)

        paddle.save(output, save_path)


def main():
    """Main function for knowledge representation learning
    """
    args = CommonArgParser().parse_args()
    args = adjust_args(args)
    set_seed(args.seed)
    set_logger(args)
    for arg in vars(args):
        logging.info('{:20}:{}'.format(arg, getattr(args, arg)))

    data = load_dataset(args.data_path, args.dataset).graph
    if args.cpu_emb:
        print(('=' * 30) + '\n using cpu embeddings\n' + ('=' * 30))
        prepare_cpu_embeddings(args, data)
    if args.filter_mode:
        pos_dict = {'head': data.pos_h4tr, 'tail': data.pos_t4hr}
    else:
        pos_dict = None

    if dist.get_world_size() > 1:
        dist.init_parallel_env()

    train_data = KGDataset(
        triplets=data.train,
        num_ents=data.num_ents,
        num_negs=args.num_negs,
        neg_mode=args.neg_mode,
        filter_mode=args.filter_mode,
        filter_dict=pos_dict)
    train_loader = DataLoader(
        dataset=train_data,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        collate_fn=train_data.mixed_collate_fn)

    if args.valid:
        assert data.valid is not None, 'validation set is not given!'
        valid_data = TestKGDataset(triplets=data.valid, num_ents=data.num_ents)
        valid_loader = DataLoader(
            dataset=valid_data,
            batch_size=args.test_batch_size,
            collate_fn=valid_data.collate_fn)

    if args.test:
        assert data.test is not None, 'test set is not given!'
        test_data = TestKGDataset(triplets=data.test, num_ents=data.num_ents)
        test_loader = DataLoader(
            dataset=test_data,
            batch_size=args.test_batch_size,
            collate_fn=test_data.collate_fn)

    model = KGEModel(
        args=args,
        num_ents=data.num_ents,
        num_rels=data.num_rels,
        score=args.score,
        embed_dim=args.embed_dim,
        ent_feat=data.ent_feat,
        rel_feat=data.rel_feat,
        ent_times=args.ent_times,
        rel_times=args.rel_times,
        scale_type=args.scale_type,
        gamma=args.gamma)
    if dist.get_world_size() > 1:
        model = paddle.DataParallel(model)

    loss_func = LossFunction(
        name=args.loss_type,
        pairwise=args.pairwise,
        margin=args.margin,
        neg_adv_spl=args.neg_adversarial_sampling,
        neg_adv_temp=args.adversarial_temperature)
    optimizer = paddle.optimizer.Adam(
        learning_rate=args.mlp_lr, parameters=model.parameters())

    timer = defaultdict(int)
    log = defaultdict(int)
    ts = t_step = time.time()
    step = 1
    for epoch in range(args.num_epoch):
        # logging.info(('=' * 10) + 'epoch %d' % epoch + ('=' * 10))

        for (h, r, t, neg_ents), all_ents, mode in train_loader:
            timer['sample'] += (time.time() - ts)

            ts = time.time()
            scores = model((h, r, t), neg_ents, all_ents, mode == 'head')
            loss = loss_func(scores['pos'], scores['neg'])
            log['loss'] += loss.numpy()[0]
            # if args.reg_coef > 0. and args.reg_norm > 0:
            #     if isinstance(model, paddle.DataParallel):
            #         params = model._layer.entity_embedding.curr_emb
            #     else:
            #         params = model.entity_embedding.curr_emb
            #     reg = paddle.norm(params, p=args.reg_norm).pow(args.reg_norm)
            #     reg = args.reg_coef * reg
            #     log['reg'] += reg.numpy()[0]
            #     loss = loss + reg
            timer['forward'] += (time.time() - ts)

            ts = time.time()
            loss.backward()
            timer['backward'] += (time.time() - ts)

            ts = time.time()
            optimizer.step()
            if args.cpu_emb:
                if isinstance(model, paddle.DataParallel):
                    model._layer.entity_embedding.step(args.lr)
                else:
                    model.entity_embedding.step(args.lr)
            optimizer.clear_grad()
            timer['update'] += (time.time() - ts)

            if (step + 1) % args.log_interval == 0:
                print_log(step, args.log_interval, log, timer, t_step)
                timer = defaultdict(int)
                log = defaultdict(int)
                t_step = time.time()

            if args.valid and (step + 1) % args.eval_interval == 0:
                test(model, valid_loader, pos_dict,
                     os.path.join(args.save_path, 'valid_%d.pkl' % (step + 1)))

            step += 1
            ts = time.time()

    if args.test:
        test(model, test_loader, pos_dict,
             os.path.join(args.save_path, 'test.pkl'))


if __name__ == '__main__':
    main()
