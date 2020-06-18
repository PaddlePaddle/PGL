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
from preprocess import get_graph_data
import pgl
import argparse
import numpy as np
import time
from paddle import fluid

import reader
from train_tool import train_epoch, valid_epoch


from model import GaANModel


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ogb Training")
    parser.add_argument("--d_name", type=str, choices=["ogbn-proteins"], default="ogbn-proteins",
                       help="the name of dataset in ogb")
    parser.add_argument("--model", type=str, choices=["GaAN"], default="GaAN",
                       help="the name of model")
    parser.add_argument("--mini_data", type=str, choices=["True", "False"], default="False",
                       help="use a small dataset to test the code")
    parser.add_argument("--use_gpu", type=bool, choices=[True, False], default=True,
                       help="use gpu")
    parser.add_argument("--gpu_id", type=int, default=0,
                       help="the id of gpu")
    parser.add_argument("--exp_id", type=int, default=0,
                       help="the id of experiment")
    parser.add_argument("--epochs", type=int, default=100,
                       help="the number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-2,
                       help="learning rate of Adam")
    parser.add_argument("--rc", type=float, default=0,
                       help="regularization coefficient")
    parser.add_argument("--log_path", type=str, default="./log",
                       help="the path of log")
    parser.add_argument("--batch_size", type=int, default=1024,
                       help="the number of batch size")
    parser.add_argument("--heads", type=int, default=8,
                       help="the number of heads of attention")
    parser.add_argument("--hidden_size_a", type=int, default=24,
                       help="the hidden size of query and key vectors")
    parser.add_argument("--hidden_size_v", type=int, default=32,
                       help="the hidden size of value vectors")
    parser.add_argument("--hidden_size_m", type=int, default=64,
                       help="the hidden size of projection for computing gates")
    parser.add_argument("--hidden_size_o", type=int ,default=128,
                       help="the hidden size of each layer in GaAN")
    
    args = parser.parse_args()

    print("Parameters Setting".center(50, "="))
    print("lr = {}, rc = {}, epochs = {}, batch_size = {}".format(args.lr, args.rc, args.epochs,
                                                                  args.batch_size))
    print("Experiment ID: {}".format(args.exp_id).center(50, "="))
    print("training in GPU: {}".format(args.gpu_id).center(50, "="))
    d_name = args.d_name
    
    # get data
    g, label, train_idx, valid_idx, test_idx, evaluator = get_graph_data(d_name=d_name, 
                                                                         mini_data=eval(args.mini_data))
    
    if args.model == "GaAN":
        graph_model = GaANModel(112, 3, args.hidden_size_a, args.hidden_size_v, args.hidden_size_m,
                                args.hidden_size_o, args.heads)
    
    # training
    samples = [25, 10] # 2-hop sample size
    batch_size = args.batch_size
    sample_workers = 1
                        
    place = fluid.CUDAPlace(args.gpu_id) if args.use_gpu else fluid.CPUPlace()           
    train_program = fluid.Program()
    startup_program = fluid.Program()

    with fluid.program_guard(train_program, startup_program):
        gw = pgl.graph_wrapper.GraphWrapper(
            name='graph',
            place = place,
            node_feat=g.node_feat_info(),
            edge_feat=g.edge_feat_info()
        )


        node_index = fluid.layers.data('node_index', shape=[None, 1], dtype="int64",
                                       append_batch_size=False)

        node_label = fluid.layers.data('node_label', shape=[None, 112], dtype="float32",
                                       append_batch_size=False)
        parent_node_index = fluid.layers.data('parent_node_index', shape=[None, 1], dtype="int64",
                                       append_batch_size=False)

        output = graph_model.forward(gw)
        output = fluid.layers.gather(output, node_index)
        score = fluid.layers.sigmoid(output)

        loss = fluid.layers.sigmoid_cross_entropy_with_logits(
            x=output, label=node_label)
        loss = fluid.layers.mean(loss)


    val_program = train_program.clone(for_test=True)

    with fluid.program_guard(train_program, startup_program):
        lr = args.lr
        adam = fluid.optimizer.Adam(
            learning_rate=lr,
            regularization=fluid.regularizer.L2DecayRegularizer(
                regularization_coeff=args.rc))
        adam.minimize(loss)

    exe = fluid.Executor(place)
    exe.run(startup_program)

    train_iter = reader.multiprocess_graph_reader(
        g,
        gw,
        samples=samples,
        num_workers=sample_workers,
        batch_size=batch_size,
        with_parent_node_index=True,
        node_index=train_idx,
        node_label=np.array(label[train_idx], dtype='float32'))

    val_iter = reader.multiprocess_graph_reader(
        g,
        gw,
        samples=samples,
        num_workers=sample_workers,
        batch_size=batch_size,
        with_parent_node_index=True,
        node_index=valid_idx,
        node_label=np.array(label[valid_idx], dtype='float32'))

    test_iter = reader.multiprocess_graph_reader(
        g,
        gw,
        samples=samples,
        num_workers=sample_workers,
        batch_size=batch_size,
        with_parent_node_index=True,
        node_index=test_idx,
        node_label=np.array(label[test_idx], dtype='float32'))


    start = time.time()
    print("Training Begin".center(50, "="))
    best_valid = -1.0
    for epoch in range(args.epochs):
        start_e = time.time()
        train_loss, train_rocauc = train_epoch(
            train_iter, program=train_program, exe=exe, loss=loss, score=score, 
            evaluator=evaluator, epoch=epoch
        )
        valid_loss, valid_rocauc = valid_epoch(
            val_iter, program=val_program, exe=exe, loss=loss, score=score,
            evaluator=evaluator, epoch=epoch)
        end_e = time.time()
        print("Epoch {}: train_loss={:.4},val_loss={:.4}, train_rocauc={:.4}, val_rocauc={:.4}, s/epoch={:.3}".format(
            epoch, train_loss, valid_loss, train_rocauc, valid_rocauc, end_e-start_e
        ))

        if valid_rocauc > best_valid:
            print("Update: new {}, old {}".format(valid_rocauc, best_valid))
            best_valid = valid_rocauc
            
            fluid.io.save_params(executor=exe, dirname='./params/'+str(args.exp_id), main_program=val_program)
            

    print("Test Stage".center(50, "="))
    
    fluid.io.load_params(executor=exe, dirname='./params/'+str(args.exp_id), main_program=val_program)
    
    test_loss, test_rocauc = valid_epoch(
        test_iter, program=val_program, exe=exe, loss=loss, score=score,
        evaluator=evaluator, epoch=epoch)
    end = time.time()
    print("test_loss={:.4},test_rocauc={:.4}, Total Time={:.3}".format(
            test_loss, test_rocauc, end-start
    ))
    print("End".center(50, "="))
