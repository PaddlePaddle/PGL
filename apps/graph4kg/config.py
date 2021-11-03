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
import math
import json
import warnings
from argparse import ArgumentParser


class KGEArgParser(ArgumentParser):
    """Argument configuration for knowledge representation learning
    """

    def __init__(self):
        super(KGEArgParser, self).__init__()

        # system
        self.add_argument('--seed', type=int, default=0, help='random seed')
        self.add_argument('--task_id', type=int, default=0, help='identifier')

        self.add_argument(
            '--data_path',
            type=str,
            default='./data/',
            help='The path of knowledge graph dataset.')

        self.add_argument(
            '--save_path',
            type=str,
            default='./output/',
            help='the path of the directory where models and logs are saved.')

        # data
        self.add_argument(
            '--data_name',
            type=str,
            default='FB15k',
            help='The name of directory where dataset files are')

        self.add_argument(
            '--batch_size',
            type=int,
            default=1000,
            help='The batch size for training.')

        self.add_argument(
            '--num_workers',
            type=int,
            default=4,
            help='num_workers for DataLoader')

        self.add_argument(
            '--neg_sample_type', type=str, default='batch', help='The range for '\
                'negative sampling. full: sampling from the whole entity set,'\
                    ' batch: sampling from entities in a batch, chunk: sampling'\
                        ' from the whole entity set as chunks')

        self.add_argument(
            '--neg_sample_size', type=int, default=1000, help='The number of '\
                'negative samples for each positive sample in the training.')

        self.add_argument(
            '--neg_deg_sample',
            action='store_true',
            help='Whether use true heads or tails to construct negative samples'
        )

        self.add_argument(
            '--filter_sample',
            action='store_true',
            help='Whether filter out true triplets in negative samples')

        self.add_argument(
            '--test_batch_size',
            type=int,
            default=50,
            help='The batch size used for validation and test.')

        self.add_argument(
            '--filter_eval',
            action='store_true',
            help='Whether filter out true triplets in evaluation candidates')

        # model
        self.add_argument(
            '--model_name',
            default='TransE',
            choices=[
                'TransE', 'RotatE', 'DistMult', 'ComplEx', 'QuatE', 'OTE'
            ])

        self.add_argument(
            '--embed_dim',
            type=int,
            default=200,
            help='The embedding size of relation and entity')

        self.add_argument(
            '--use_feature',
            action='store_true',
            help='Whether use feature embedding and feed [feature, emb] into mlp'
        )

        self.add_argument(
            '-adv',
            '--neg_adversarial_sampling',
            action='store_true',
            help='Indicate whether to use negative adversarial sampling.'\
                    'It will weight negative samples with higher scores more.')

        self.add_argument(
            '-a',
            '--adversarial_temperature',
            default=1.0,
            type=float,
            help='The temperature used for negative adversarial sampling.')

        self.add_argument(
            '-rc',
            '--reg_coef',
            type=float,
            default=0.000002,
            help='The coefficient for regularization.')

        self.add_argument(
            '-rn',
            '--reg_norm',
            type=int,
            default=3,
            help='norm used in regularization.')

        self.add_argument(
            '--loss_type',
            default='Logsigmoid',
            choices=['Hinge', 'Logistic', 'Logsigmoid', 'BCE', 'Softplus'],
            help='The loss function used to train KGE Model.')

        self.add_argument(
            '-m',
            '--margin',
            type=float,
            default=1.0,
            help='The margin value in Hinge loss.')

        self.add_argument(
            '-pw',
            '--pairwise',
            action='store_true',
            help='Indicate whether to use pairwise loss function.')

        # optional parameters for score functions
        self.add_argument(
            '-g',
            '--gamma',
            type=float,
            default=12.0,
            help='The margin value in the score function.')

        self.add_argument('--ote_scale_type', type=int, default=0)

        self.add_argument('--ote_size', type=int, default=1)

        # traning
        self.add_argument(
            '--num_epoch',
            type=int,
            default=1000000,
            help='The maximal number of epochs to train.')

        self.add_argument(
            '--lr', type=float, default=0.01, help='The learning rate.')

        self.add_argument(
            '--mlp_lr',
            type=float,
            default=0.0001,
            help='The learning rate to optimize non-embeddings')

        self.add_argument(
            '--mix_cpu_gpu',
            action='store_true',
            help='Whether use cpu embedding')

        self.add_argument(
            '--async_update',
            action='store_true',
            help='Allow asynchronous update on node embedding for multi-GPU training.'\
                                  'This overlaps CPU and GPU computation to speed up.')

        self.add_argument('--print_on_screen', action='store_true')

        self.add_argument(
            '-log', '--log_interval', type=int, default=1000, help='Print'\
                ' runtime of different components every x steps.')

        self.add_argument(
            '--valid', action='store_true', help='Evaluate the model on'\
                ' the validation set during training.')

        self.add_argument(
            '--test', action='store_true', help='Evaluate the model on '\
                'the test set after the model is trained.')

        self.add_argument(
            '--eval_interval', type=int, default=50000, help='Print '\
                'evaluation results on the validation dataset every x steps')


def load_model_config(config_file):
    """Load configuration from config.yaml
    """
    with open(config_file, "r") as f:
        config = json.loads(f.read())
    return config


def prepare_save_path(args):
    """Create save path and makedirs if not exists
    """
    task_name = '{}_{}_d_{}_g_{}_e_{}_r_{}_l_{}_lr_{}_{}_{}'.format(
        args.model_name, args.data_name, args.embed_dim, args.gamma, 'cpu'
        if args.ent_emb_on_cpu else 'gpu', 'cpu' if args.rel_emb_on_cpu else
        'gpu', args.loss_type, args.lr, args.mlp_lr, args.task_id)

    args.save_path = os.path.join(args.save_path, task_name)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    else:
        warnings.warn('save path {} exists, it will be overwriten.'.format(
            args.save_path))

    return args


def prepare_data_config(args):
    """Adjust configuration for data processing
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
        args.num_chunks = args.batch_size // args.neg_sample_size
    else:
        args.num_chunks = 1

    return args


def prepare_embedding_config(args):
    """Specify configuration of embeddings
    """
    # device
    mix_cpu_on_relation = False
    args.rel_emb_on_cpu = args.mix_cpu_gpu and mix_cpu_on_relation
    args.ent_emb_on_cpu = args.mix_cpu_gpu

    print(('-' * 40) + '\n        Device Setting        \n' + ('-' * 40))
    ent_place = 'cpu' if args.ent_emb_on_cpu else 'gpu'
    rel_place = 'cpu' if args.rel_emb_on_cpu else 'gpu'
    print(' Entity   embedding place: {}'.format(ent_place))
    print(' Relation embedding place: {}'.format(rel_place))
    print(('-' * 40))

    # dimension
    if args.model_name == 'rotate':
        args.ent_dim = args.embed_dim * 2
        args.rel_dim = args.embed_dim
    elif args.model_name == 'ote':
        args.ent_dim = args.embed_dim
        args.rel_dim = args.embed_dim * (
            args.ote_size + int(args.ote_scale_type > 0))
    else:
        args.ent_dim = args.embed_dim
        args.rel_dim = args.embed_dim
    print('-' * 40 + '\n       Embedding Setting      \n' + ('-' * 40))
    print(' Entity   embedding dimension: {}'.format(args.ent_dim))
    print(' Relation embedding dimension: {}'.format(args.rel_dim))
    print(('-' * 40))
    return args


def prepare_model_config(args):
    """Standardizing str arguments
    """
    args.model_name = args.model_name.lower()
    if args.async_update:
        print('=' * 20 + '\n Async Update!\n' + '=' * 20)
    if args.async_update and not args.mix_cpu_gpu:
        raise ValueError("We only support async_update in mix_cpu_gpu mode.")

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
