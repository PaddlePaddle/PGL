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
import pickle
import logging
import time

from models.base_model import BaseKEModel
from dataloader import TrainDataset, EvalDataset, NewBidirectionalOneShotIterator
from dataloader import get_dataset
from utils import get_compatible_batch_size, CommonArgParser, load_model_config

import paddle
import random
from math import ceil
from tqdm import tqdm
import numpy as np
from collections import defaultdict
from ogb.lsc import WikiKG90MDataset, WikiKG90MEvaluator


class ArgParser(CommonArgParser):
    def __init__(self):
        super(ArgParser, self).__init__()

        self.add_argument(
            '--infer_valid',
            action='store_true',
            help='Evaluate the model on the validation set in the training.')
        self.add_argument(
            '--infer_test',
            action='store_true',
            help='Evaluate the model on the validation set in the training.')
        self.add_argument(
            '--model_path',
            type=str,
            default='ckpts',
            help='the place where to load the model.')


def use_config_replace_args(args, config):
    """
    Update the running config to the args
    """
    for key, value in config.items():
        if key != "data_path":
            setattr(args, key, value)
    return args


def load_model_from_checkpoint(model, model_path):
    """
    Load the MLP and entity parameters from the checkpoint
    """
    entity_weight = np.load(
        os.path.join(model_path, "entity_params.npy"), mmap_mode="r")
    model.entity_embedding.weight = entity_weight
    model.relation_embedding.set_state_dict(
        paddle.load(os.path.join(model_path, "relation.pdparams")))
    model.transform_net.set_state_dict(
        paddle.load(os.path.join(model_path, "mlp.pdparams")))


def infer(args, model, config, rank, samplers, mode="valid"):
    with paddle.no_grad():
        logs = defaultdict(list)
        answers = defaultdict(list)
        scores = defaultdict(list)
        for sampler in samplers:
            for query, ans, candidate in tqdm(
                    sampler,
                    total=ceil(sampler.num_edges / sampler.batch_size)):
                log, score = model.forward_test_wikikg(query, ans, candidate,
                                                       sampler.mode)
                logs[sampler.mode].append(log)
                answers[sampler.mode].append(ans)
                scores[sampler.mode].append(score)
        input_dict = {}
        if mode == "valid":
            assert len(answers) == 1
            assert 'h,r->t' in answers
            if 'h,r->t' in answers:
                assert 'h,r->t' in logs, "h,r->t not in logs"
                input_dict['h,r->t'] = {
                    't_correct_index': paddle.concat(
                        answers['h,r->t'], axis=0).numpy(),
                    't_pred_top10': paddle.concat(
                        logs['h,r->t'], axis=0).numpy()
                }
                input_dict['h,r->t']['scores'] = paddle.concat(
                    scores["h,r->t"], axis=0).numpy()
        else:
            input_dict['h,r->t'] = {
                't_pred_top10': paddle.concat(logs['h,r->t'], 0).numpy()
            }
            input_dict['h,r->t']['scores'] = paddle.concat(scores["h,r->t"],
                                                           0).numpy()
        with open(
                os.path.join(args.model_path,
                             "{}_{}_0.pkl".format(mode, rank)), "wb") as f:
            pickle.dump(input_dict, f)


def main():
    """
    Main predict function for the wikikg90m
    """
    args = ArgParser().parse_args()
    config = load_model_config(
        os.path.join(args.model_path, 'model_config.json'))
    args = use_config_replace_args(args, config)
    dataset = get_dataset(args, args.data_path, args.dataset, args.format,
                          args.delimiter, args.data_files,
                          args.has_edge_importance)
    print("Load the dataset done.")
    eval_dataset = EvalDataset(dataset, args)

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
        cpu_emb=args.cpu_emb,
        relation_times=args.ote_size,
        scale_type=args.scale_type)

    print("Create the model done.")
    model.entity_feat = dataset.entity_feat
    model.relation_feat = dataset.relation_feat
    load_model_from_checkpoint(model, args.model_path)
    print("The model load the checkpoint done.")

    if args.infer_valid:
        valid_sampler_tail = eval_dataset.create_sampler(
            'valid',
            args.batch_size_eval,
            mode='tail',
            num_workers=args.num_workers,
            rank=0,
            ranks=1)
        infer(args, model, config, 0, [valid_sampler_tail], "valid")

    if args.infer_test:
        test_sampler_tail = eval_dataset.create_sampler(
            'test',
            args.batch_size_eval,
            mode='tail',
            num_workers=args.num_workers,
            rank=i,
            ranks=args.num_proc)
        infer(args, model, config, 0, [test_sampler_tail], "test")


if __name__ == "__main__":
    main()
