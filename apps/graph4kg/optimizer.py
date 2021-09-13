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

import paddle
from paddle.optimizer import Adam, SGD


class NumpySgdOptimizer(paddle.optimizer.SGD):
    def __init__(self, *args, **kwargs):
        self._config = locals()
        self.np_layers = kwargs.pop("numpy_layer", None)

        super().__init__(*args, **kwargs)

    def step(self, *args, **kwargs):
        if self.np_layers is not None:
            for idx, np_layer in enumerate(self.np_layers):
                numpy_grads_list = {}
                for k, v in np_layer.tensor_name_dict.items():
                    numpy_grads_list[k] = (
                        -self._learning_rate *
                        v.grad).numpy()  #self.state_dict()[v].grad
                    if idx == 0:
                        pass
                        #print(self._learning_rate, np.mean(np.abs(v.grad.numpy())), np.mean(v.grad.numpy()))
                np_layer.update_sgd(numpy_grads_list)
        if len(self._param_groups) == 0:
            return None
        return super().step(*args, **kwargs)

    def clear_grad(self, *args, **kwargs):
        if len(self._param_groups) == 0:
            return None
        return super().clear_grad(*args, **kwargs)


class NumpyAdagradOptimizer(paddle.optimizer.Adagrad):
    def __init__(self, *args, **kwargs):
        self._config = locals()
        self.np_layers = kwargs.pop("numpy_layer", None)

        super().__init__(*args, **kwargs)

    def step(self, *args, **kwargs):
        if self.np_layers is not None:
            for idx, np_layer in enumerate(self.np_layers):
                numpy_grads_list = {}
                for k, v in np_layer.tensor_name_dict.items():
                    numpy_grads_list[k] = v.grad
                np_layer.update_adagrad(numpy_grads_list, self._learning_rate)

#         if len(self._param_groups) == 0:
#             return None
        return super().step(*args, **kwargs)

    def clear_grad(self, *args, **kwargs):
        #         if len(self._param_groups) == 0:
        #             return None
        return super().clear_grad(*args, **kwargs)
