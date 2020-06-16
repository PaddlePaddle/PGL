# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""finetune args"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import

import os
import time
import argparse

from utils.args import ArgumentGroup

# yapf: disable
parser = argparse.ArgumentParser(__doc__)
model_g = ArgumentGroup(parser, "model", "model configuration and paths.")
model_g.add_arg("init_checkpoint", str, None, "Init checkpoint to resume training from.")
model_g.add_arg("init_pretraining_params", str, None,
 "Init pre-training params which preforms fine-tuning from. If the "
 "arg 'init_checkpoint' has been set, this argument wouldn't be valid.")

train_g = ArgumentGroup(parser, "training", "training options.")
train_g.add_arg("epoch", int, 3, "Number of epoches for fine-tuning.")
train_g.add_arg("learning_rate", float, 5e-5, "Learning rate used to train with warmup.")

run_type_g = ArgumentGroup(parser, "run_type", "running type options.")
run_type_g.add_arg("use_cuda", bool, True, "If set, use GPU for training.")
run_type_g.add_arg("num_workers", int, 4, "use multiprocess to generate graph")
run_type_g.add_arg("output_path", str, None, "path to save model")
run_type_g.add_arg("model", str,  None, "model to run")
run_type_g.add_arg("hidden_size", int, 256, "model hidden-size")
run_type_g.add_arg("drop_rate", float, 0.5, "Dropout rate")
run_type_g.add_arg("batch_size", int, 1024, "batch_size")
run_type_g.add_arg("full_batch", bool, False, "use static graph wrapper, if full_batch is true, batch_size will take no effect.")
run_type_g.add_arg("samples", type=int, nargs='+', default=[30, 30], help="sample nums of k-hop.")
run_type_g.add_arg("test_batch_size", int, 512, help="sample nums of k-hop of test phase.")
run_type_g.add_arg("test_samples", type=int, nargs='+', default=[30, 30], help="sample nums of k-hop.")
