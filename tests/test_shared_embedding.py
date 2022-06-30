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

import time
import unittest

import numpy as np
import paddle

from pgl.utils.shared_embedding import SharedEmbedding


class TestSharedEmebdding(unittest.TestCase):
    def setUp(self):
        self.weight_path = 'test_default_embedding.npy'
        self.x = np.array([[0., 3.], [5., 9.], [1., 7.]])

    def create_random_shared_embedding(self):
        embeds = SharedEmbedding(
            num_embeddings=5, embedding_dim=10, weight_path=self.weight_path)
        return embeds

    def test_create_shared_embedding(self):
        u_embeds = self.create_random_shared_embedding()
        v_embeds = SharedEmbedding.from_file(self.weight_path)
        v_embeds.weight[0][0] = 15
        self.assertEqual(u_embeds.weight[0][0], 15)

    def test_create_from_array(self):
        embeds = SharedEmbedding.from_array(
            self.x, weight_path='test_weight_embedding.npy')
        self.assertTrue((embeds.weight == self.x).all())

    def test_train_mode(self):
        embeds = self.create_random_shared_embedding()
        embeds.train()
        x = embeds(np.array([0, 2]))
        self.assertFalse(x.stop_gradient)

    def test_eval_mode(self):
        embeds = self.create_random_shared_embedding()
        embeds.eval()
        x = embeds(np.array([0, 2]))
        self.assertTrue(x.stop_gradient)

    def test_property_weight_path(self):
        embeds = self.create_random_shared_embedding()
        self.assertEqual(embeds.weight_path, self.weight_path)

    def test_property_current_embedding(self):
        embeds = self.create_random_shared_embedding()
        indexs = paddle.to_tensor([0, 2])

        tensors = embeds(indexs)
        curr_embs = embeds.curr_emb
        self.assertTrue((tensors == curr_embs).all())

    def test_get(self):
        embeds = SharedEmbedding.from_array(self.x, self.weight_path)
        sub_embs = embeds.get(np.array([0, 2]))
        self.assertTrue((embeds.weight[[0, 2]] == sub_embs).all())

    def test_create_trace(self):
        embeds = SharedEmbedding.from_array(self.x, self.weight_path)
        u = paddle.to_tensor([[1.], [-1]])
        indexs = paddle.to_tensor([0, 2])
        grad = paddle.to_tensor([[1., 1.], [-1, -1]])

        tensors = embeds(indexs)
        loss = (tensors * u).sum()
        loss.backward()
        trace = embeds.create_trace(paddle.to_tensor(indexs), tensors)
        self.assertTrue((trace[1][0] == grad).all())

    def test_create_trace_none(self):
        embeds = SharedEmbedding.from_array(self.x, self.weight_path)
        indexs = paddle.to_tensor([0, 2])

        trace = embeds.create_trace(indexs, None)
        self.assertIsNone(trace)

    def test_step_trace(self):
        embeds = SharedEmbedding.from_array(self.x, self.weight_path,
                                            'adagrad', 1.)
        indexs = paddle.to_tensor([0, 2])
        grad = [
            paddle.ones([2, 2]).astype('float32'),
            paddle.ones([2, ]).astype('float32')
        ]
        embeds.step_trace([indexs, grad])
        y = np.array([[-1., 2.], [5., 9.], [0., 6.]])
        self.assertTrue((embeds.weight == y).all())

    def test_async_update(self):
        try:
            embeds = SharedEmbedding.from_array(self.x, self.weight_path,
                                                'adagrad', 1.)
            embeds.start_async_update()
            u = paddle.to_tensor([[1.], [-1]])
            indexs = paddle.to_tensor([0, 2])
            tensors = embeds(indexs)
            loss = (tensors * u).sum()
            loss.backward()
            embeds.step()
            y = np.array([[-1., 2.], [5., 9.], [2., 8.]])
            time.sleep(5)
            self.assertTrue((embeds.weight == y).all())
            embeds.finish_async_update()
        except AssertionError as error:
            embeds.finish_async_update()
            raise AssertionError(error)


if __name__ == '__main__':
    unittest.main()
