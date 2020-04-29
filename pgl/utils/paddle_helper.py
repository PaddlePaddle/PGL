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
paddle_helper package contain some simple function to help building
paddle models.
"""
import warnings
import numpy as np

import paddle
from paddle.fluid import core
import paddle.fluid as fluid
import paddle.fluid.layer_helper as layer_helper
from pgl.utils.logger import log


def gather(input, index):
    """Gather input from given index.

    Slicing input data with given index. This function rewrite paddle.fluid.layers.gather
    to fix issue: https://github.com/PaddlePaddle/Paddle/issues/17509 when paddlepaddle's
    version is less than 1.5.

    Args:
        input: Input tensor to be sliced

        index: Slice index

    Return:
        A tensor that are sliced from given input data.
    """
    try:
        # PaddlePaddle 1.5
        output = fluid.layers.gather(input, index, overwrite=False)
        return output
    except TypeError as e:
        warnings.warn("Your paddle version is less than 1.5"
                      " gather may be slower.")

        if index.dtype == core.VarDesc.VarType.INT32:
            index = fluid.layers.cast(index, "int64")
            if index.shape[-1] != 1:
                index = fluid.layers.reshape(index, shape=[-1, 1])
            index.stop_gradient = True

        helper = layer_helper.LayerHelper("gather", **locals())  #**locals())
        dtype = input.dtype
        tmp = helper.create_variable_for_type_inference(dtype)
        padding_idx = -1
        helper.append_op(
            type='lookup_table',
            inputs={'Ids': index,
                    'W': input},
            outputs={'Out': tmp},
            attrs={
                'is_sparse': False,
                'is_distributed': False,
                'remote_prefetch': False,
                'padding_idx': padding_idx
            })
        return tmp


def constant(name, value, dtype, hide_batch_size=True):
    """Create constant variable with given data.

    This function helps to create constants variable with
    given numpy.ndarray data.

    Args:
        name: variable name

        value: numpy.ndarray the value of constant

        dtype: the type of constant

        hide_batch_size: If set the first dimenstion as unknown, the explicit
                         batch size may cause some error in paddle. For example,
                         when the value has a shape of (batch_size, dim1, dim2),
                         it will return a variable with shape (-1, dim1, dim2).

    Return:
        A tuple contain the constant variable and the constant
        variable initialize function.

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            place = fluid.CPUPlace()
            exe = fluid.Executor(place)
            constant_var, constant_var_init = constant(name="constant",
                              value=np.array([5.0],
                              dtype="float32"))
            exe.run(fluid.default_startup_program())
            # Run After default startup
            constant_var_init(place)

    """
    if not isinstance(value, np.ndarray):
        raise TypeError("value should be Numpy array.")

    value = value.astype(dtype)
    data = fluid.layers.create_global_var(
        shape=value.shape,
        value=0,
        dtype=value.dtype,
        name=name,
        persistable=True)
    data.stop_gradient = True

    if hide_batch_size:
        shape = list(value.shape)
        shape[0] = -1
        data.desc.set_shape(shape)

    def initializer(place):
        if isinstance(place, fluid.CUDAPlace):
            pass
        elif isinstance(place, fluid.CUDAPinnedPlace):
            pass
        elif isinstance(place, fluid.CPUPlace):
            pass
        else:
            raise TypeError(
                "The input of initializer is not in"
                " [fluid.CUDAPlace, fluid.CPUPlace, fluid.CUDAPinnedPlace]")
        var = fluid.global_scope().var(data.name).get_tensor()
        var.set(value, place)

    return data, initializer


def lod_constant(name, value, lod, dtype):
    """Create constant lod variable with given data,

    This function helps to create constants lod variable with given numpy.ndarray data
    and lod information.

    Args:
        name: variable name

        value: numpy.ndarray the value of constant

        dtype: the type of constant

        lod: lod infos of given value.

    Return:
        A tuple contain the constant variable and the constant
        variable initialize function.

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            place = fluid.CPUPlace()
            exe = fluid.Executor(place)
            constant_var, constant_var_init = lod_constant(name="constant",
                              value=np.array([[5.0], [1.0], [2.0]],
                              lod=[2, 1],
                              dtype="float32"))
            exe.run(fluid.default_startup_program())
            # Run After default startup
            constant_var_init(place)
    """
    data, data_initializer = constant(
        name=name, value=value, dtype=dtype, hide_batch_size=True)

    _lod = [0]
    for l in lod:
        _lod.append(_lod[-1] + l)
    output = fluid.layers.lod_reset(data, target_lod=_lod)
    return output, data_initializer


def sequence_softmax(x):
    """Compute sequence softmax over paddle LodTensor

    This function compute softmax normalization along with the length of sequence.
    This function is an extention of :code:`fluid.layers.sequence_softmax` which can only
    deal with LodTensor whose last dimension is 1.

    Args:
        x: The input variable which is a LodTensor.

    Return:
        Output of sequence_softmax
    """
    x_max = fluid.layers.sequence_pool(x, "max")
    x_max = fluid.layers.sequence_expand_as(x_max, x)
    x = x - x_max
    exp_x = fluid.layers.exp(x)
    sum_exp_x = fluid.layers.sequence_pool(exp_x, "sum")
    sum_exp_x = fluid.layers.sequence_expand_as(sum_exp_x, exp_x)
    return exp_x / sum_exp_x


def scatter_add(input, index, updates):
    """Scatter add updates to input by given index.

    Adds sparse updates to input variables.

    Args:
        input: Input tensor to be updated

        index: Slice index

        updates: Must have same type as input.

    Return:
        Same type and shape as input.
    """

    output = fluid.layers.scatter(input, index, updates, overwrite=False)
    return output


def scatter_max(input, index, updates):
    """Scatter max updates to input by given index.

    Adds sparse updates to input variables.

    Args:
        input: Input tensor to be updated

        index: Slice index

        updates: Must have same type as input.

    Return:
        Same type and shape as input.
    """

    output = fluid.layers.scatter(input, index, updates, mode='max')
    return output
