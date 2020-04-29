# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved
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
"""
This file preprocess the data before training.
"""

import sys
import argparse


def gen_nodes_file(file_, result_file):
    """calculate the total number of nodes and save them for latter processing.
    """
    nodes = []
    with open(file_, 'r') as reader:
        for line in reader:
            tokens = line.strip().split(' ')
            nodes.append(tokens[1])
            nodes.append(tokens[2])

    nodes = list(set(nodes))
    nodes.sort(key=int)
    print('total number of nodes: %d' % len(nodes))
    print('saving nodes file in %s' % (result_file))
    with open(result_file, 'w') as writer:
        for n in nodes:
            writer.write(n + '\n')

    print('finished')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='GATNE')
    parser.add_argument(
        '--input_file',
        default='./data/youtube/train.txt',
        type=str,
        help='input file')
    parser.add_argument(
        '--output_file',
        default='./data/youtube/nodes.txt',
        type=str,
        help='output file')
    args = parser.parse_args()

    print('generating nodes file')
    gen_nodes_file(args.input_file, args.output_file)
