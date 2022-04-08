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
import argparse
import logging
import numpy as np
import paddle
import torch

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# yapf: disable
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default='FB15k', help="The name of dataset.")
parser.add_argument("--init_from_ckpt", type=str, default='', help="The path of checkpoint to be loaded.")
parser.add_argument("--model_name", type=str, default='TransE', help="The model name that converting the parameter.")
parser.add_argument("--mode", type=str, default='pgl2dgl', help="The name that from the source framework to destination framework.")
parser.add_argument("--save_path", type=str, default='', help="The save directory path for the parameter converting.")
args = parser.parse_args()
# yapf: enable


def model_pgl2dgl(args):
    state_dict = paddle.load(
        os.path.join(args.init_from_ckpt, "params.pdparams"))
    if args.model_name == 'TransE':
        dgl_model_name = 'TransE_l2'
    else:
        dgl_model_name = args.model_name
    mapped_dict = {
        'ent_embedding.weight':
        args.dataset + '_' + dgl_model_name + '_entity.npy',
        'rel_embedding.weight':
        args.dataset + '_' + dgl_model_name + '_relation.npy'
    }
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    for key in mapped_dict.keys():
        parameter = state_dict[key].numpy()
        parameter_file_name = mapped_dict[key]
        with open(os.path.join(args.save_path, parameter_file_name),
                  'wb') as f:
            np.save(f, parameter)

    logger.info("The PGL parameter convert to DGL parameter success.")


def model_dgl2pgl(args):
    if args.model_name == 'TransE':
        dgl_model_name = 'TransE_l2'
    else:
        dgl_model_name = args.model_name
    embedding_file = os.path.join(
        args.init_from_ckpt,
        args.dataset + '_' + dgl_model_name + '_entity.npy')
    relation_file = os.path.join(
        args.init_from_ckpt,
        args.dataset + '_' + dgl_model_name + '_relation.npy')
    embedding_tensor = paddle.to_tensor(np.load(embedding_file))
    relation_tensor = paddle.to_tensor(np.load(relation_file))
    state_dict = {
        'ent_embedding.weight': embedding_tensor,
        'rel_embedding.weight': relation_tensor
    }
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    paddle.save(state_dict, os.path.join(args.save_path, 'params.pdparams'))
    logger.info("The DGL parameter convert to PGL parameter success.")


def main(args):
    if args.mode == 'pgl2dgl':
        model_pgl2dgl(args)
    elif args.mode == 'dgl2pgl':
        model_dgl2pgl(args)
    else:
        raise ("The mode is not correct, just could be pgl2dgl/dgl2pgl.")


if __name__ == "__main__":
    main(args)
