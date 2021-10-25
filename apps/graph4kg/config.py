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

from argparse import ArgumentParser


class KGEArgParser(ArgumentParser):
    """Argument configuration for knowledge representation learning
    """

    def __init__(self):
        super(KGEArgParser, self).__init__()

        # input
        self.add_argument(
            '--data_path',
            type=str,
            default='~/data/',
            help='The path of knowledge graph dataset.')
        self.add_argument(
            '--data_name',
            type=str,
            default='FB15k',
            help='The name of directory where dataset files are')
        self.add_argument('--seed', type=int, default=0, help='random seed')
        self.add_argument(
            '--use_feature',
            action='store_true',
            help='Whether use feature embedding')
        self.add_argument('--train_percent', type=float, default=1.0)
        self.add_argument(
            '--eval_percent',
            type=float,
            default=0.1,
            help='Randomly sample some percentage of edges for evaluation.')
        self.add_argument(
            '--test_percent',
            type=float,
            default=1.0,
            help='Randomly sample some percentage of edges for test.')

        # output
        self.add_argument(
            '--save_path',
            type=str,
            default='ckpts',
            help='the path of the directory where models and logs are saved.')
        self.add_argument(
            '--save_step',
            type=int,
            default=100000,
            help='The step where models and logs are saved.')
        self.add_argument(
            '--no_save_emb',
            action='store_true',
            help='Disable saving embeddings under save_path.')
        self.add_argument('--print_on_screen', action='store_true')
        self.add_argument(
            '--save_threshold',
            type=float,
            default=0.85,
            help='save threshold for mrr.')

        # device
        self.add_argument(
            '--mix_cpu_gpu',
            action='store_true',
            help='Whether use cpu embedding')

        # sampler
        self.add_argument(
            '--batch_size',
            type=int,
            default=1000,
            help='The batch size for training.')
        self.add_argument(
            '--test_batch_size',
            type=int,
            default=50,
            help='The batch size used for validation and test.')
        self.add_argument(
            '--neg_sample_size', type=int, default=1000, help='The number of '\
                'negative samples for each positive sample in the training.')
        self.add_argument(
            '--neg_sample_type', type=str, default='batch', help='The range for '\
                'negative sampling. full: sampling from the whole entity set,'\
                    ' batch: sampling from entities in a batch, chunk: sampling'\
                        ' from the whole entity set as chunks')
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
            '--filter_eval',
            action='store_true',
            help='Whether filter out true triplets in evaluation candidates')
        self.add_argument(
            '--num_workers',
            type=int,
            default=4,
            help='num_workers for DataLoader')
        self.add_argument('-adv', '--neg_adversarial_sampling', action='store_true',
                          help='Indicate whether to use negative adversarial sampling.'\
                                  'It will weight negative samples with higher scores more.')
        self.add_argument(
            '-a',
            '--adversarial_temperature',
            default=1.0,
            type=float,
            help='The temperature used for negative adversarial sampling.')

        # model
        self.add_argument(
            '--num_epoch',
            type=int,
            default=1000000,
            help='The maximal number of epochs to train.')
        self.add_argument(
            '--valid', action='store_true', help='Evaluate the model on'\
                ' the validation set during training.')
        self.add_argument(
            '--test', action='store_true', help='Evaluate the model on '\
                'the test set after the model is trained.')
        self.add_argument(
            '--no_eval_filter',
            action='store_true',
            help='Disable filter positive edges from randomly constructed negative edges for evaluation'
        )
        self.add_argument(
            '--eval_interval', type=int, default=50000, help='Print '\
                'evaluation results on the validation dataset every x steps')
        self.add_argument(
            '-log', '--log_interval', type=int, default=1000, help='Print'\
                ' runtime of different components every x steps.')
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
            '--num_proc', type=int, default=1, help='The number of processes '\
                'to train the model in parallel. In multi-GPU training, the '\
                    'number of processes by default is set to match the '\
                        'number of GPUs. If set explicitly, the number of '\
                            'processes needs to be divisible by the number of GPUs.')
        self.add_argument('--num_thread', type=int, default=1,
                          help='The number of CPU threads to train the model in each process.'\
                                  'This argument is used for multiprocessing training.')
        self.add_argument('--async_update', action='store_true',
                          help='Allow asynchronous update on node embedding for multi-GPU training.'\
                                  'This overlaps CPU and GPU computation to speed up.')
        self.add_argument(
            '--force_sync_interval', type=int, default=-1, help='We force a '\
                'synchronization between processes every x steps for '\
                    'multiprocessing training. This potentially stablizes '\
                        'the training process to get a better performance. '\
                            'For multiprocessing training, it is set to 1000 '\
                                ' by default.')

        # optimizer
        self.add_argument(
            '--lr', type=float, default=0.01, help='The learning rate.')
        self.add_argument(
            '--mlp_lr',
            type=float,
            default=0.0001,
            help='The learning rate to optimize non-embeddings')
        self.add_argument('--lr_decay_rate', type=float, default=None)
        self.add_argument('--lr_decay_interval', type=int, default=10000)
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
            '-pw', '--pairwise', action='store_true', help='Indicate whether '\
                'to use pairwise loss function.')

        # score function
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
        self.add_argument('--ent_times', type=int, default=1)
        self.add_argument('--rel_times', type=int, default=1)
        self.add_argument(
            '-g',
            '--gamma',
            type=float,
            default=12.0,
            help='The margin value in the score function.')
        self.add_argument('--scale_type', type=int, default=0)
        self.add_argument('--ote_size', type=int, default=1)
