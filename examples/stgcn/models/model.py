# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
"""This file implement the STGCN model.
"""
import numpy as np

import paddle.fluid as fluid
import paddle.fluid.layers as fl
import pgl


class STGCNModel(object):
    """Implementation of Spatio-Temporal Graph Convolutional Networks"""

    def __init__(self, args, gw):
        self.args = args
        self.gw = gw

        self.input = fl.data(
            name="input",
            shape=[None, args.n_his + 1, args.n_route, 1],
            dtype="float32")

    def forward(self):
        """forward"""
        x = self.input[:, 0:self.args.n_his, :, :]
        # Ko>0: kernel size of temporal convolution in the output layer.
        Ko = self.args.n_his
        # ST-Block
        for i, channels in enumerate(self.args.blocks):
            x = self.st_conv_block(
                x,
                self.args.Ks,
                self.args.Kt,
                channels,
                "st_conv_%d" % i,
                self.args.keep_prob,
                act_func='GLU')

        # output layer
        if Ko > 1:
            y = self.output_layer(x, Ko, 'output_layer')
        else:
            raise ValueError(f'ERROR: kernel size Ko must be greater than 1, \
                    but received "{Ko}".')

        label = self.input[:, self.args.n_his:self.args.n_his + 1, :, :]
        train_loss = fl.reduce_sum((y - label) * (y - label))
        single_pred = y[:, 0, :, :]  # shape: [batch, n, 1]

        return train_loss, single_pred

    def st_conv_block(self,
                      x,
                      Ks,
                      Kt,
                      channels,
                      name,
                      keep_prob,
                      act_func='GLU'):
        """Spatio-Temporal convolution block"""
        c_si, c_t, c_oo = channels

        x_s = self.temporal_conv_layer(
            x, Kt, c_si, c_t, "%s_tconv_in" % name, act_func=act_func)
        x_t = self.spatio_conv_layer(x_s, Ks, c_t, c_t, "%s_sonv" % name)
        x_o = self.temporal_conv_layer(x_t, Kt, c_t, c_oo,
                                       "%s_tconv_out" % name)

        x_ln = fl.layer_norm(x_o)
        return fl.dropout(x_ln, dropout_prob=(1.0 - keep_prob))

    def temporal_conv_layer(self, x, Kt, c_in, c_out, name, act_func='relu'):
        """Temporal convolution layer"""
        _, T, n, _ = x.shape
        if c_in > c_out:
            x_input = fl.conv2d(
                input=x,
                num_filters=c_out,
                filter_size=[1, 1],
                stride=[1, 1],
                padding="SAME",
                data_format="NHWC",
                param_attr=fluid.ParamAttr(name="%s_conv2d_1" % name))
        elif c_in < c_out:
            # if the size of input channel is less than the output,
            # padding x to the same size of output channel.
            pad = fl.fill_constant_batch_size_like(
                input=x,
                shape=[-1, T, n, c_out - c_in],
                dtype="float32",
                value=0.0)
            x_input = fl.concat([x, pad], axis=3)
        else:
            x_input = x

        #  x_input = x_input[:, Kt - 1:T, :, :]
        if act_func == 'GLU':
            # gated liner unit
            bt_init = fluid.initializer.ConstantInitializer(value=0.0)
            bt = fl.create_parameter(
                shape=[2 * c_out],
                dtype="float32",
                attr=fluid.ParamAttr(
                    name="%s_bt" % name, trainable=True, initializer=bt_init),
            )
            x_conv = fl.conv2d(
                input=x,
                num_filters=2 * c_out,
                filter_size=[Kt, 1],
                stride=[1, 1],
                padding="SAME",
                data_format="NHWC",
                param_attr=fluid.ParamAttr(name="%s_conv2d_wt" % name))
            x_conv = x_conv + bt
            return (x_conv[:, :, :, 0:c_out] + x_input
                    ) * fl.sigmoid(x_conv[:, :, :, -c_out:])
        else:
            bt_init = fluid.initializer.ConstantInitializer(value=0.0)
            bt = fl.create_parameter(
                shape=[c_out],
                dtype="float32",
                attr=fluid.ParamAttr(
                    name="%s_bt" % name, trainable=True, initializer=bt_init),
            )
            x_conv = fl.conv2d(
                input=x,
                num_filters=c_out,
                filter_size=[Kt, 1],
                stride=[1, 1],
                padding="SAME",
                data_format="NHWC",
                param_attr=fluid.ParamAttr(name="%s_conv2d_wt" % name))
            x_conv = x_conv + bt
            if act_func == "linear":
                return x_conv
            elif act_func == "sigmoid":
                return fl.sigmoid(x_conv)
            elif act_func == "relu":
                return fl.relu(x_conv + x_input)
            else:
                raise ValueError(
                    f'ERROR: activation function "{act_func}" is not defined.')

    def spatio_conv_layer(self, x, Ks, c_in, c_out, name):
        """Spatio convolution layer"""
        _, T, n, _ = x.shape
        if c_in > c_out:
            x_input = fl.conv2d(
                input=x,
                num_filters=c_out,
                filter_size=[1, 1],
                stride=[1, 1],
                padding="SAME",
                data_format="NHWC",
                param_attr=fluid.ParamAttr(name="%s_conv2d_1" % name))
        elif c_in < c_out:
            # if the size of input channel is less than the output,
            # padding x to the same size of output channel.
            pad = fl.fill_constant_batch_size_like(
                input=x,
                shape=[-1, T, n, c_out - c_in],
                dtype="float32",
                value=0.0)
            x_input = fl.concat([x, pad], axis=3)
        else:
            x_input = x

        for i in range(Ks):
            # x_input shape: [B,T, num_nodes, c_out]
            x_input = fl.reshape(x_input, [-1, c_out])

            x_input = self.message_passing(
                self.gw,
                x_input,
                name="%s_mp_%d" % (name, i),
                norm=self.gw.node_feat["norm"])

            x_input = fl.fc(x_input,
                            size=c_out,
                            bias_attr=False,
                            param_attr=fluid.ParamAttr(name="%s_gcn_fc_%d" %
                                                       (name, i)))

            bias = fluid.layers.create_parameter(
                shape=[c_out],
                dtype='float32',
                is_bias=True,
                name='%s_gcn_bias_%d' % (name, i))
            x_input = fluid.layers.elementwise_add(x_input, bias, act="relu")

            x_input = fl.reshape(x_input, [-1, T, n, c_out])

        return x_input

    def message_passing(self, gw, feature, name, norm=None):
        """Message passing layer"""

        def send_src_copy(src_feat, dst_feat, edge_feat):
            """send function"""
            return src_feat["h"] * edge_feat['w']

        if norm is not None:
            feature = feature * norm

        msg = gw.send(
            send_src_copy,
            nfeat_list=[("h", feature)],
            efeat_list=[('w', gw.edge_feat['weights'])])
        output = gw.recv(msg, "sum")

        if norm is not None:
            output = output * norm

        return output

    def output_layer(self, x, T, name, act_func='GLU'):
        """Output layer"""
        _, _, n, channel = x.shape

        # maps multi-steps to one.
        x_i = self.temporal_conv_layer(
            x=x,
            Kt=T,
            c_in=channel,
            c_out=channel,
            name="%s_in" % name,
            act_func=act_func)
        x_ln = fl.layer_norm(x_i)
        x_o = self.temporal_conv_layer(
            x=x_ln,
            Kt=1,
            c_in=channel,
            c_out=channel,
            name="%s_out" % name,
            act_func='sigmoid')

        # maps multi-channels to one.
        x_fc = self.fully_con_layer(
            x=x_o, n=n, channel=channel, name="%s_fc" % name)
        return x_fc

    def fully_con_layer(self, x, n, channel, name):
        """Fully connected layer"""
        bt_init = fluid.initializer.ConstantInitializer(value=0.0)
        bt = fl.create_parameter(
            shape=[n, 1],
            dtype="float32",
            attr=fluid.ParamAttr(
                name="%s_bt" % name, trainable=True, initializer=bt_init), )
        x_conv = fl.conv2d(
            input=x,
            num_filters=1,
            filter_size=[1, 1],
            stride=[1, 1],
            padding="SAME",
            data_format="NHWC",
            param_attr=fluid.ParamAttr(name="%s_conv2d" % name))
        x_conv = x_conv + bt
        return x_conv
