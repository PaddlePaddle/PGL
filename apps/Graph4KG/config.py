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
import math
import json
import warnings
from argparse import ArgumentParser

import paddle.distributed as dist


class KGEArgParser(ArgumentParser):
    """Argument configuration for knowledge representation learning
    """

    def __init__(self):
        super(KGEArgParser, self).__init__()
        self.basic_group = self.add_argument_group('basic',
                                                   'required arguments.')

        self.basic_group.add_argument(
            '--seed',
            type=int,
            default=0,
            help='Random seed for initialization.')

        self.basic_group.add_argument(
            '--data_path',
            type=str,
            default='./data/',
            help='Directory of knowledge graph dataset.')

        self.basic_group.add_argument(
            '--save_path',
            type=str,
            default='./output/',
            help='Directory to save model and log.')

        self.basic_group.add_argument(
            '--init_from_ckpt',
            type=str,
            default=None,
            help='Directory to load the model.')

        self.basic_group.add_argument(
            '--data_name',
            type=str,
            default='FB15k',
            choices=[
                'FB15k', 'FB15k-237', 'wn18', 'WN18RR', 'wikikg2', 'wikikg90m'
            ],
            help='Dataset name.')

        self.basic_group.add_argument(
            '--use_dict',
            type=bool,
            default=False,
            help='Use the dict to index the data.')

        self.basic_group.add_argument(
            '--kv_mode',
            type=str,
            default='kv',
            help='The order of string names and ids in dictionary files. kv denotes entity_name/relation_name, id.'
        )

        self.basic_group.add_argument(
            '--batch_size',
            type=int,
            default=1000,
            help='Number of triplets in a batch for training.')

        self.basic_group.add_argument(
            '--test_batch_size',
            type=int,
            default=16,
            help='Number of triplets in a batch for validation and test.')

        self.basic_group.add_argument(
            '--neg_sample_size',
            type=int,
            default=1,
            help='Number of negative samples of each triplet for training.')

        self.basic_group.add_argument(
            '--filter_eval',
            action='store_true',
            help='Filter out existing triplets from evaluation candidates.')

        self.basic_group.add_argument(
            '--model_name',
            default='TransE',
            choices=[
                'TransE', 'RotatE', 'DistMult', 'ComplEx', 'QuatE', 'OTE'
            ],
            help='Knowledge embedding method for training.')

        self.basic_group.add_argument(
            '--embed_dim',
            type=int,
            default=200,
            help='Dimension of real entity and relation embeddings.')

        self.basic_group.add_argument(
            '-rc',
            '--reg_coef',
            type=float,
            default=0,
            help='Coefficient of regularization.')

        self.basic_group.add_argument(
            '--loss_type',
            default='Logsigmoid',
            choices=['Hinge', 'Logistic', 'Logsigmoid', 'BCE', 'Softplus'],
            help='Loss function of KGE Model.')

        self.basic_group.add_argument(
            '--max_steps',
            type=int,
            default=2000000,
            help='Number of batches to train.')

        self.basic_group.add_argument(
            '--lr',
            type=float,
            default=0.1,
            help='Learning rate to optimize model parameters.')

        self.basic_group.add_argument(
            '--optimizer',
            type=str,
            default='adagrad',
            choices=['adam', 'adagrad', 'sgd'],
            help='Optimizer of model parameters.')

        self.basic_group.add_argument(
            '--cpu_lr',
            type=float,
            default=0.1,
            help='Learning rate to optimize shared embeddings on CPU.')

        self.basic_group.add_argument(
            '--cpu_optimizer',
            type=str,
            default='adagrad',
            choices=['sgd', 'adagrad'],
            help='Optimizer of shared embeddings on CPU.')

        self.basic_group.add_argument(
            '--mix_cpu_gpu',
            action='store_true',
            help='Use shared embeddings and store entity embeddings on CPU.')

        self.basic_group.add_argument(
            '--async_update',
            action='store_true',
            help='Asynchronously update embeddings with gradients.')

        self.basic_group.add_argument(
            '--valid', action='store_true', help='Evaluate the model on'\
                ' the validation set during training.')

        self.basic_group.add_argument(
            '--test', action='store_true', help='Evaluate the model on '\
                'the test set after the model is trained.')

        self.data_group = self.add_argument_group('data optional')

        self.data_group.add_argument(
            '--task_name', type=str, default='KGE', help='Task identifier.')

        self.data_group.add_argument(
            '--num_workers',
            type=int,
            default=0,
            help='Number of workers used to load batch data.')

        self.data_group.add_argument(
            '--neg_sample_type',
            type=str,
            default='chunk',
            choices=['chunk', 'full', 'batch'],
            help='The type of negative sampling. \n"chunk": sampled from all '\
                'entities; triplets are devided into several chunks and each '\
                    'chunk shares a group of negative samples.\n"full": sampled'\
                        ' from all entities.\n"batch": sampling from current batch.\n')

        self.data_group.add_argument(
            '--neg_deg_sample',
            action='store_true',
            help='Use true heads or tails in negative sampling. See details in'\
                'https://arxiv.org/abs/1902.10197.')

        self.data_group.add_argument(
            '-adv',
            '--neg_adversarial_sampling',
            action='store_true',
            help='Use negative adversarial sampling, which weights '\
                'negative samples with higher scores more.')

        self.data_group.add_argument(
            '-a',
            '--adversarial_temperature',
            default=1.0,
            type=float,
            help='Temperature used for negative adversarial sampling.')

        self.data_group.add_argument(
            '--filter_sample',
            action='store_true',
            help='Filter out existing triplets in negative samples.')

        self.data_group.add_argument(
            '--valid_percent',
            type=float,
            default=1.,
            help='Percent of used validation triplets.')

        self.model_group = self.add_argument_group('model optional')

        self.model_group.add_argument(
            '--use_feature',
            action='store_true',
            help='Use features for training.')

        self.model_group.add_argument(
            '-rt',
            '--reg_type',
            type=str,
            default='norm_er',
            choices=['norm_er', 'norm_hrt'],
            help='Regularization type.\n"norm_er": compute norm of '\
                'entities and relations seperately.\n"norm_hrt": '\
                    'compute norm of heads, relations and tails seperately.')

        self.model_group.add_argument(
            '-rn',
            '--reg_norm',
            type=int,
            default=3,
            help='Order of regularization norm.')

        self.model_group.add_argument(
            '--weighted_loss',
            action='store_true',
            help='Use weights of samples when computing loss. See details in'\
                'https://arxiv.org/abs/1902.10197.')

        self.model_group.add_argument(
            '-m',
            '--margin',
            type=float,
            default=1.0,
            help='Margin value in Hinge loss.')

        self.model_group.add_argument(
            '-pw',
            '--pairwise',
            action='store_true',
            help='Compute pairwise loss of triplets and negative samples.')

        self.opt_group = self.add_argument_group('score function optional')
        self.opt_group.add_argument(
            '-g',
            '--gamma',
            type=float,
            default=12.0,
            help='Margin value of triplet scores.')

        self.opt_group.add_argument(
            '--ote_scale',
            type=int,
            default=0,
            choices=[0, 1, 2],
            help='Scale method in OTE. 0-None; 1-abs; 2-exp.')

        self.opt_group.add_argument(
            '--ote_size',
            type=int,
            default=1,
            help='Number of linear transform matrix in OTE.')

        self.opt_group.add_argument(
            '--quate_lmbda1',
            type=float,
            default=0.,
            help='Coefficient of the first regularization in QuatE.')

        self.opt_group.add_argument(
            '--quate_lmbda2',
            type=float,
            default=0.,
            help='Coefficient of the second regularization in QuatE.')

        self.train_group = self.add_argument_group('train optional')
        self.train_group.add_argument(
            '--num_epoch',
            type=int,
            default=1000000,
            help='Number of epochs to train.')

        self.train_group.add_argument(
            '--scheduler_interval',
            type=int,
            default=-1,
            help='Interval size to update learning rate of model. -1 denotes constant.'
        )

        self.train_group.add_argument(
            '--num_process',
            type=int,
            default=1,
            help='Number of processes for asynchroneous gradient update.')

        self.train_group.add_argument(
            '--print_on_screen',
            action='store_true',
            help='Print logs in console.')

        self.train_group.add_argument(
            '-log', '--log_interval', type=int, default=1000, help='Print'\
                ' runtime of different components every x steps.')

        self.train_group.add_argument(
            '--save_interval',
            type=int,
            default=-1,
            help='Interval size to save model checkpoint.')

        self.train_group.add_argument(
            '--eval_interval', type=int, default=50000, help='Print '\
                'evaluation results on the validation dataset every x steps.')


def load_model_config(config_file):
    """Load configuration from config.yaml.
    """
    with open(config_file, "r") as f:
        config = json.loads(f.read())
    return config


def prepare_save_path(args):
    """Create save path and makedirs if not exists.
    """
    task_name = '{}_{}_d_{}_g_{}_e_{}_r_{}_l_{}_lr_{}_{}_{}'.format(
        args.model_name, args.data_name, args.embed_dim, args.gamma, 'cpu'
        if args.ent_emb_on_cpu else 'gpu', 'cpu' if args.rel_emb_on_cpu else
        'gpu', args.loss_type, args.lr, args.cpu_lr, args.task_name)

    args.save_path = os.path.join(args.save_path, task_name)

    if dist.get_rank() == 0:
        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path)
    return args


def prepare_data_config(args):
    """Adjust configuration for data processing.
    """
    batch_size = args.batch_size
    neg_sample_size = args.neg_sample_size
    neg_sample_type = args.neg_sample_type
    if neg_sample_type == 'chunk' and neg_sample_size < batch_size:
        if batch_size % neg_sample_size != 0:
            batch_size = int(
                math.ceil(batch_size / neg_sample_size) * neg_sample_size)
            print('For "chunk" negative sampling, batch size should ' \
                'be divisible by negative sample size {}. Thus, ' \
                    'batch_size {} is reset as {}'.format(neg_sample_size,
                        args.batch_size, batch_size))
            args.batch_size = batch_size

    if neg_sample_type == 'chunk':
        args.num_chunks = max(args.batch_size // args.neg_sample_size, 1)
    else:
        args.num_chunks = args.batch_size

    return args


def prepare_embedding_config(args):
    """Specify configuration of embeddings.
    """
    # Device
    args.ent_emb_on_cpu = args.mix_cpu_gpu
    # As the number of relations in KGs is relatively small, we put relation
    # emebddings on GPUs by default to speed up training.
    args.rel_emb_on_cpu = False

    print(('-' * 40) + '\n        Device Setting        \n' + ('-' * 40))
    ent_place = 'cpu' if args.ent_emb_on_cpu else 'gpu'
    rel_place = 'cpu' if args.rel_emb_on_cpu else 'gpu'
    print(' Entity   embedding place: {}'.format(ent_place))
    print(' Relation embedding place: {}'.format(rel_place))
    print(('-' * 40))

    return args


def prepare_model_config(args):
    """Standardizing str arguments.
    """
    args.model_name = args.model_name.lower()
    if args.async_update:
        print('=' * 20 + '\n Async Update!\n' + '=' * 20)
    if args.async_update and not args.mix_cpu_gpu:
        raise ValueError("We only support async_update in mix_cpu_gpu mode.")
    if args.reg_coef > 0:
        assert args.reg_norm >= 0, 'norm of regularization is negative!'
    if args.reg_type == 'norm_er':
        args.use_embedding_regularization = args.reg_coef > 0
    else:
        args.use_embedding_regularization = (args.quate_lmbda1 > 0) \
            or (args.quate_lmbda2 > 0)

    # Dimension
    if args.model_name == 'rotate':
        args.ent_dim = args.embed_dim * 2
        args.rel_dim = args.embed_dim
    elif args.model_name == 'complex':
        args.ent_dim = args.embed_dim * 2
        args.rel_dim = args.embed_dim * 2
    elif args.model_name == 'quate':
        args.ent_dim = args.embed_dim * 4
        args.rel_dim = args.embed_dim * 4
    elif args.model_name == 'ote':
        args.ent_dim = args.embed_dim
        args.rel_dim = args.embed_dim * (
            args.ote_size + int(args.ote_scale > 0))
    else:
        args.ent_dim = args.embed_dim
        args.rel_dim = args.embed_dim
    print('-' * 40 + '\n       Embedding Setting      \n' + ('-' * 40))
    print(' Entity   embedding dimension: {}'.format(args.ent_dim))
    print(' Relation embedding dimension: {}'.format(args.rel_dim))
    print(('-' * 40))
    return args


def prepare_config():
    """Load arguments and preprocess them
    """
    args = KGEArgParser().parse_args()
    args = prepare_embedding_config(args)
    args = prepare_model_config(args)
    args = prepare_save_path(args)
    args = prepare_data_config(args)
    return args
