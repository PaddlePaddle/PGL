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
import time
import os
role = os.getenv("TRAINING_ROLE", "TRAINER")

import numpy as np
from pgl.utils.logger import log
from pgl.utils.log_writer import LogWriter
import paddle.fluid as F
import paddle.fluid.layers as L
from paddle.fluid.incubate.fleet.parameter_server.distribute_transpiler import StrategyFactory
from paddle.fluid.incubate.fleet.collective import DistributedStrategy
from paddle.fluid.transpiler.distribute_transpiler import DistributeTranspilerConfig
from paddle.fluid.incubate.fleet.collective import fleet as cfleet
from paddle.fluid.incubate.fleet.parameter_server.distribute_transpiler import fleet as tfleet
import paddle.fluid.incubate.fleet.base.role_maker as role_maker
from paddle.fluid.transpiler.distribute_transpiler import DistributedMode
from paddle.fluid.incubate.fleet.parameter_server.distribute_transpiler.distributed_strategy import TrainerRuntimeConfig

# hack it!
base_get_communicator_flags = TrainerRuntimeConfig.get_communicator_flags
def get_communicator_flags(self):
    flag_dict = base_get_communicator_flags(self)
    flag_dict['communicator_max_merge_var_num'] = str(1)
    flag_dict['communicator_send_queue_size'] = str(1)
    return flag_dict
TrainerRuntimeConfig.get_communicator_flags = get_communicator_flags


class Learner(object):
    @classmethod
    def factory(cls, name):
        if name == "cpu":
            return TranspilerLearner()
        elif name == "gpu":
            return CollectiveLearner()
        else:
            raise ValueError

    def build(self, model, data_gen, config):
        raise NotImplementedError

    def warmstart(self, program, path='./checkpoints'):
        def _existed_persitables(var):
            #if not isinstance(var, fluid.framework.Parameter):
            #    return False
            if not F.io.is_persistable(var):
                return False
            param_path = os.path.join(path, var.name)
            log.info("Loading parameter: {} persistable: {} exists: {}".format(
                param_path,
                F.io.is_persistable(var),
                os.path.exists(param_path),
            ))
            return os.path.exists(param_path)
        F.io.load_vars(
            self.exe,
            path,
            main_program=program,
            predicate=_existed_persitables
        )

    def start(self):
        batch = 0
        start = time.time()
        trainer_id = int(os.getenv("PADDLE_TRAINER_ID", "0"))
        if trainer_id == 0:
            writer = LogWriter(os.path.join(self.config.output_path, "train_history"))

        for epoch_idx in range(self.config.epoch):
            for idx, batch_feed_dict in enumerate(self.model.data_loader()):
                try:
                    cpu_time = time.time()
                    batch += 1
                    batch_loss  = self.exe.run(
                        self.program,
                        feed=batch_feed_dict,
                        fetch_list=[self.model.loss])
                    end = time.time()
                    if trainer_id == 0:
                        writer.add_scalar("loss", np.mean(batch_loss), batch)
                        if batch % self.config.log_per_step == 0:
                            log.info(
                                "Epoch %s Batch %s %s-Loss %s \t Speed(per batch) %.5lf/%.5lf sec"
                                % (epoch_idx, batch, "train", np.mean(batch_loss), (end - start) /batch, (end - cpu_time)))
                            writer.flush()
                        if batch % self.config.save_per_step == 0:
                            self.fleet.save_persistables(self.exe, os.path.join(self.config.output_path, str(batch)))
                except Exception as e:
                    log.info("Pyreader train error")
                    log.exception(e)
            log.info("epcoh %s done." % epoch_idx)

    def stop(self):
        self.fleet.save_persistables(self.exe, os.path.join(self.config.output_path, "last"))


class TranspilerLearner(Learner):
    def __init__(self):
        training_role = os.getenv("TRAINING_ROLE", "TRAINER")
        paddle_role = role_maker.Role.WORKER
        place = F.CPUPlace()
        if training_role == "PSERVER":
            paddle_role = role_maker.Role.SERVER

        # set the fleet runtime environment according to configure
        port = os.getenv("PADDLE_PORT", "6174")
        pserver_ips = os.getenv("PADDLE_PSERVERS")  # ip,ip...
        eplist = []
        for ip in pserver_ips.split(","):
            eplist.append(':'.join([ip, port]))
        pserver_endpoints = eplist  # ip:port,ip:port...
        worker_num = int(os.getenv("PADDLE_TRAINERS_NUM", "0"))
        trainer_id = int(os.getenv("PADDLE_TRAINER_ID", "0"))
        role = role_maker.UserDefinedRoleMaker(
            current_id=trainer_id,
            role=paddle_role,
            worker_num=worker_num,
            server_endpoints=pserver_endpoints)
        tfleet.init(role)
        tfleet.save_on_pserver = True

    def build(self, model, data_gen, config):
        self.optimize(model.loss, config.optimizer_type, config.lr)
        self.init_and_run_ps_worker(config.ckpt_path)
        self.program = self.complie_program(model.loss)
        self.fleet = tfleet
        model.data_loader.decorate_batch_generator(
            data_gen, places=F.cpu_places())
        self.config = config
        self.model = model

    def optimize(self, loss, optimizer_type, lr):
        log.info('learning rate:%f' % lr)
        if optimizer_type == "sgd":
            optimizer = F.optimizer.SGD(learning_rate=lr)
        elif optimizer_type == "adam":
            # Don't slice tensor ensure convergence 
            optimizer = F.optimizer.Adam(learning_rate=lr, lazy_mode=True)
        else:
            raise ValueError("Unknown Optimizer %s" % optimizer_type)
        #create the DistributeTranspiler configure
        self.strategy = StrategyFactory.create_sync_strategy()
        optimizer = tfleet.distributed_optimizer(optimizer, self.strategy)
        optimizer.minimize(loss)

    def init_and_run_ps_worker(self, ckpt_path):
        # init and run server or worker
        self.exe = F.Executor(F.CPUPlace())
        if tfleet.is_server():
            tfleet.init_server()
            self.warmstart(tfleet.startup_program, path=ckpt_path)
            tfleet.run_server()
            exit()

        if tfleet.is_worker():
            log.info("start init worker done")
            tfleet.init_worker()
            self.exe.run(tfleet.startup_program)

    def complie_program(self, loss):
        num_threads = int(os.getenv("CPU_NUM", 1))
        exec_strategy = F.ExecutionStrategy()
        exec_strategy.num_threads = num_threads
        exec_strategy.use_thread_barrier = False
        build_strategy = F.BuildStrategy()
        build_strategy.enable_inplace = True
        build_strategy.memory_optimize = True
        build_strategy.remove_unnecessary_lock = False
        build_strategy.memory_optimize = False
        build_strategy.async_mode = False

        if num_threads > 1:
            build_strategy.reduce_strategy = F.BuildStrategy.ReduceStrategy.Reduce

        log.info("start build compile program...")
        compiled_prog = F.compiler.CompiledProgram(tfleet.main_program
            ).with_data_parallel(
                loss_name=loss.name,
                build_strategy=build_strategy,
                exec_strategy=exec_strategy)

        return compiled_prog


class CollectiveLearner(Learner):
    def __init__(self):
        role = role_maker.PaddleCloudRoleMaker(is_collective=True)
        cfleet.init(role)

    def optimize(self, loss, optimizer_type, lr):
        optimizer = F.optimizer.Adam(learning_rate=lr)
        dist_strategy = DistributedStrategy()
        dist_strategy.enable_sequential_execution = True
        optimizer = cfleet.distributed_optimizer(optimizer, strategy=dist_strategy)
        _, param_grads = optimizer.minimize(loss, F.default_startup_program())
    
    def build(self, model, data_gen, config):
        self.optimize(model.loss, config.optimizer_type, config.lr)
        self.program = cfleet.main_program
        gpu_id = int(os.getenv("FLAGS_selected_gpus", "0"))
        place = F.CUDAPlace(gpu_id)
        self.exe = F.Executor(place)
        self.exe.run(F.default_startup_program())
        self.warmstart(F.default_startup_program(), config.ckpt_path)
        self.fleet = cfleet
        model.data_loader.decorate_batch_generator(
            data_gen, places=place)
        self.config = config
        self.model = model
