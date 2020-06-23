# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved
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

parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=777, 
                    help='seed')
parser.add_argument('--batch_size', type=int, default=128,
                    help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.0005,
                    help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.0001,
                    help='weight decay')
parser.add_argument('--hidden_size', type=int, default=128,
                    help='gcn hidden size')
parser.add_argument('--pooling_ratio', type=float, default=0.5,
                    help='pooling ratio of SAGPool')
parser.add_argument('--dropout_ratio', type=float, default=0.5,
                    help='dropout ratio')
parser.add_argument('--dataset_name', type=str, default='DD',
                    help='DD/PROTEINS/NCI1/NCI109/FRANKENSTEIN')
parser.add_argument('--epochs', type=int, default=100000,
                    help='maximum number of epochs')
parser.add_argument('--patience', type=int, default=50,
                    help='patience for early stopping')
parser.add_argument('--use_cuda', type=bool, default=True,
                    help='use cuda or cpu')
parser.add_argument('--save_model', type=str,  
                   help='save model name')

