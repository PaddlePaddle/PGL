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
Utils for the models.
"""
import paddle.fluid as fluid
from paddle.fluid.layer_helper import LayerHelper


def lookup_table(input, embedding_table, dtype='float32'):
    """
    lookup table support for paddle.
    :param input:
    :param embedding_table:
    :param dtype:
    :return:
    """
    is_sparse = False
    is_distributed = False
    helper = LayerHelper('embedding', **locals())
    remote_prefetch = is_sparse and (not is_distributed)
    if remote_prefetch:
        assert is_sparse is True and is_distributed is False
    tmp = helper.create_variable_for_type_inference(dtype)
    padding_idx = -1
    helper.append_op(
        type='lookup_table',
        inputs={'Ids': input,
                'W': embedding_table},
        outputs={'Out': tmp},
        attrs={
            'is_sparse': is_sparse,
            'is_distributed': is_distributed,
            'remote_prefetch': remote_prefetch,
            'padding_idx': padding_idx
        })
    return tmp


def lookup_table_gather(index, input):
    """
    lookup table support for paddle by gather.
    :param index:
    :param input:
    :return:
    """
    return fluid.layers.gather(index=index, input=input, overwrite=False)


def _clone_var_in_block_(block, var):
    assert isinstance(var, fluid.Variable)
    if var.desc.type() == fluid.core.VarDesc.VarType.LOD_TENSOR:
        return block.create_var(
            name=var.name,
            shape=var.shape,
            dtype=var.dtype,
            type=var.type,
            lod_level=var.lod_level,
            persistable=True)
    else:
        return block.create_var(
            name=var.name,
            shape=var.shape,
            dtype=var.dtype,
            type=var.type,
            persistable=True)


def load_var(executor, main_program=None, var=None, filename=None):
    """
    load_var to certain program
    :param executor: executor
    :param main_program: the program to load
    :param var: the variable name in main_program.
    :file_name: the file name of the file to load.
    :return: None
    """
    load_prog = fluid.Program()
    load_block = load_prog.global_block()

    if main_program is None:
        main_program = fluid.default_main_program()

    if not isinstance(main_program, fluid.Program):
        raise TypeError("program should be as Program type or None")

    vars = list(filter(None, main_program.list_vars()))
    # save origin param shape
    orig_para_shape = {}
    load_var_map = {}
    for each_var in vars:
        if each_var.name != var:
            continue
        assert isinstance(each_var, fluid.Variable)
        if each_var.type == fluid.core.VarDesc.VarType.RAW:
            continue

        if isinstance(each_var, fluid.framework.Parameter):
            orig_para_shape[each_var.name] = tuple(each_var.desc.get_shape())
        new_var = _clone_var_in_block_(load_block, each_var)
        if filename is not None:
            load_block.append_op(
                type='load',
                inputs={},
                outputs={'Out': [new_var]},
                attrs={'file_path': filename})

    executor.run(load_prog)
