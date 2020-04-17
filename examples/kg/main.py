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
The script to run these models.
"""
import argparse
import timeit
import paddle.fluid as fluid
from data_loader import KBloader
from evalutate import Evaluate
from model import model_dict
from mp_mapper import mp_reader_mapper
from pgl.utils.logger import log


def run_round(batch_iter,
              program,
              exe,
              fetch_list,
              epoch,
              prefix="train",
              log_per_step=1000):
    """
    Run the program for one epoch.
    :param batch_iter: the batch_iter of prepared data.
    :param program: the running program, train_program or test program.
    :param exe: the executor of paddle.
    :param fetch_list: the variables to fetch.
    :param epoch: the epoch number of train process.
    :param prefix: the prefix name, type `string`.
    :param log_per_step: log per step.
    :return: None
    """
    batch = 0
    tmp_epoch = 0
    loss = 0
    tmp_loss = 0
    run_time = 0
    data_time = 0
    t2 = timeit.default_timer()
    for batch_feed_dict in batch_iter():
        batch += 1
        t1 = timeit.default_timer()
        data_time += (t1 - t2)
        batch_fetch = exe.run(program,
                              fetch_list=fetch_list,
                              feed=batch_feed_dict)
        if prefix == "train":
            loss += batch_fetch[0]
            tmp_loss += batch_fetch[0]
        if batch % log_per_step == 0:
            tmp_epoch += 1
            if prefix == "train":
                log.info("Epoch %s Ava Loss %s" %
                         (epoch + tmp_epoch, tmp_loss / batch))
            else:
                log.info("Batch %s" % batch)
            batch = 0
            tmp_loss = 0

        t2 = timeit.default_timer()
        run_time += (t2 - t1)

    if prefix == "train":
        log.info("GPU run time {}, Data prepare extra time {}".format(
            run_time, data_time))
        log.info("Epoch %s \t All Loss %s" % (epoch + tmp_epoch, loss))


def train(args):
    """
    Train the knowledge graph embedding model.
    :param args: all args.
    :return: None
    """
    kgreader = KBloader(
        batch_size=args.batch_size,
        data_dir=args.data_dir,
        neg_mode=args.neg_mode,
        neg_times=args.neg_times)
    if args.model in model_dict:
        Model = model_dict[args.model]
    else:
        raise ValueError("No model for name {}".format(args.model))
    model = Model(
        data_reader=kgreader,
        hidden_size=args.hidden_size,
        margin=args.margin,
        learning_rate=args.learning_rate,
        args=args,
        optimizer=args.optimizer)

    def iter_map_wrapper(data_batch, repeat=1):
        """
        wrapper for multiprocess reader
        :param data_batch: the source data iter.
        :param repeat: repeat data for multi epoch
        :return: iterator of feed data
        """

        def data_repeat():
            """repeat data for multi epoch"""
            for i in range(repeat):
                for d in data_batch():
                    yield d

        reader = mp_reader_mapper(
            data_repeat,
            func=kgreader.training_data_map,
            #func=kgreader.training_data_no_filter,
            num_works=args.sample_workers)

        return reader

    def iter_wrapper(data_batch, feed_list):
        """
        Decorator of make up the feed dict
        :param data_batch: the source data iter.
        :param feed_list: the feed list (names of variables).
        :return: iterator of feed data.
        """

        def work():
            """work"""
            for batch in data_batch():
                feed_dict = {}
                for k, v in zip(feed_list, batch):
                    feed_dict[k] = v
                yield feed_dict

        return work

    loader = fluid.io.DataLoader.from_generator(
        feed_list=model.train_feed_vars, capacity=20, iterable=True)

    places = fluid.cuda_places() if args.use_cuda else fluid.cpu_places()
    exe = fluid.Executor(places[0])
    exe.run(model.startup_program)
    exe.run(fluid.default_startup_program())

    prog = fluid.CompiledProgram(model.train_program).with_data_parallel(
        loss_name=model.train_fetch_vars[0].name)

    if args.only_evaluate:
        s = timeit.default_timer()
        fluid.io.load_params(
            exe, dirname=args.checkpoint, main_program=model.train_program)
        Evaluate(kgreader).launch_evaluation(
            exe=exe,
            reader=iter_wrapper(kgreader.test_data_batch,
                                model.test_feed_list),
            fetch_list=model.test_fetch_vars,
            program=model.test_program,
            num_workers=10)
        log.info(timeit.default_timer() - s)
        return None

    batch_iter = iter_map_wrapper(
        kgreader.training_data_batch,
        repeat=args.evaluate_per_iteration, )
    loader.set_batch_generator(batch_iter, places=places)

    for epoch in range(0, args.epoch // args.evaluate_per_iteration):
        run_round(
            batch_iter=loader,
            exe=exe,
            prefix="train",
            # program=model.train_program,
            program=prog,
            fetch_list=model.train_fetch_vars,
            log_per_step=kgreader.train_num // args.batch_size,
            epoch=epoch * args.evaluate_per_iteration)
        log.info("epoch\t%s" % ((1 + epoch) * args.evaluate_per_iteration))
        if True:
            fluid.io.save_params(
                exe, dirname=args.checkpoint, main_program=model.train_program)
            eva = Evaluate(kgreader)
            eva.launch_evaluation(
                exe=exe,
                reader=iter_wrapper(kgreader.test_data_batch,
                                    model.test_feed_list),
                fetch_list=model.test_fetch_vars,
                program=model.test_program,
                num_workers=10)


def main():
    """
    The main entry of all.
    :return: None
    """
    parser = argparse.ArgumentParser(
        description="Knowledge Graph Embedding for PGL")
    parser.add_argument('--use_cuda', action='store_true', help="use_cuda")
    parser.add_argument(
        '--data_dir',
        dest='data_dir',
        type=str,
        help='the directory of dataset',
        default='./data/WN18/')
    parser.add_argument(
        '--model',
        dest='model',
        type=str,
        help="model to run",
        default="TransE")
    parser.add_argument(
        '--learning_rate',
        dest='learning_rate',
        type=float,
        help='learning rate',
        default=0.001)
    parser.add_argument(
        '--epoch', dest='epoch', type=int, help='epoch to run', default=400)
    parser.add_argument(
        '--sample_workers',
        dest='sample_workers',
        type=int,
        help='sample workers',
        default=4)
    parser.add_argument(
        '--batch_size',
        dest='batch_size',
        type=int,
        help="batch size",
        default=1000)
    parser.add_argument(
        '--optimizer',
        dest='optimizer',
        type=str,
        help='optimizer',
        default='adam')
    parser.add_argument(
        '--hidden_size',
        dest='hidden_size',
        type=int,
        help='embedding dimension',
        default=50)
    parser.add_argument(
        '--margin', dest='margin', type=float, help='margin', default=4.0)
    parser.add_argument(
        '--checkpoint',
        dest='checkpoint',
        type=str,
        help='directory to save checkpoint directory',
        default='output/')
    parser.add_argument(
        '--evaluate_per_iteration',
        dest='evaluate_per_iteration',
        type=int,
        help='evaluate the training result per x iteration',
        default=50)
    parser.add_argument(
        '--only_evaluate',
        dest='only_evaluate',
        action='store_true',
        help='only do the evaluate program',
        default=False)
    parser.add_argument(
        '--adv_temp_value', type=float, help='adv_temp_value', default=2.0)
    parser.add_argument('--neg_times', type=int, help='neg_times', default=1)
    parser.add_argument(
        '--neg_mode', type=bool, help='return neg mode flag', default=False)

    args = parser.parse_args()
    log.info(args)
    train(args)


if __name__ == '__main__':
    main()
