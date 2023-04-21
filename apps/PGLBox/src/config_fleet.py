# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
"""Generate Proto for pslib
"""
import os
import copy
import numpy as np
import paddle
from pgl.utils.logger import log
from paddle.distributed.fleet.base.role_maker import PaddleCloudRoleMaker
from paddle.distributed.fleet.base.role_maker import Role, Gloo
from multiprocessing import Manager, Process


def get_strategy(args, model_dict):
    strategy = paddle.distributed.fleet.DistributedStrategy()
    strategy.a_sync = True  # 默认使用async模式

    configs = {"use_ps_gpu": 1}
    strategy.a_sync_configs = configs

    # sparse参数相关配置
    strategy.fleet_desc_configs = generate_config(args)

    if args.fs_name or args.fs_ugi:
        user, passwd = args.fs_ugi.split(',', 1)
        strategy.fs_client_param = {
            "uri": args.fs_name,
            "user": user,
            "passwd": passwd,
            "hadoop_bin": "%s/bin/hadoop" % (os.getenv("HADOOP_HOME"))
        }
        log.info("set DistributedStrategy fs_client_param")
    else:
        strategy.fs_client_param = {
            "uri": "",
            "user": "",
            "passwd": "",
            "hadoop_bin": ""
        }
        log.info("not set DistributedStrategy fs_client_param")

    return strategy


def gen_sparse_config(args, sparse_lr, init_range, op_type, emb_size,
                      feature_lr, nodeid_slot, load_filter_slots,
                      sparse_table_class):
    """
    gen sparse config
    """
    sparse_config = dict()
    sparse_config['sparse_table_class'] = sparse_table_class
    sparse_config['sparse_compress_in_save'] = True
    sparse_config['sparse_shard_num'] = 67
    # sparse_config['sparse_accessor_class'] = "DownpourCtrAccessor"
    sparse_config[
        'sparse_accessor_class'] = "DownpourCtrDymfAccessor"  # for variable embedding
    sparse_config['sparse_learning_rate'] = sparse_lr
    sparse_config['sparse_initial_g2sum'] = 3
    sparse_config['sparse_initial_range'] = init_range
    sparse_config['sparse_weight_bounds'] = [-10.0, 10.0]
    sparse_config['sparse_embedx_dim'] = emb_size
    sparse_config['sparse_embedx_threshold'] = 0
    sparse_config['sparse_nonclk_coeff'] = 0.1
    sparse_config['sparse_click_coeff'] = 1.0
    sparse_config['sparse_base_threshold'] = 0
    sparse_config['sparse_delta_threshold'] = 0.25
    sparse_config['sparse_delta_keep_days'] = 16.0
    sparse_config['sparse_show_click_decay_rate'] = 0.98
    sparse_config['sparse_delete_threshold'] = 0.8
    sparse_config['sparse_delete_after_unseen_days'] = 30

    sparse_config['embed_sparse_optimizer'] = op_type
    sparse_config['embed_sparse_learning_rate'] = sparse_lr
    sparse_config['embed_sparse_initial_range'] = 0
    sparse_config[
        'embed_sparse_beta1_decay_rate'] = 0.9  #args.beta1_decay_rate
    sparse_config[
        'embed_sparse_beta2_decay_rate'] = 0.999  #args.beta2_decay_rate
    sparse_config['embed_sparse_weight_bounds'] = [-10.0, 10.0]

    sparse_config['embedx_sparse_optimizer'] = op_type
    sparse_config['embedx_sparse_learning_rate'] = sparse_lr
    sparse_config['embedx_sparse_initial_range'] = init_range
    sparse_config[
        'embedx_sparse_beta1_decay_rate'] = 0.9  #args.beta1_decay_rate
    sparse_config[
        'embedx_sparse_beta2_decay_rate'] = 0.999  #args.beta2_decay_rate
    sparse_config['embedx_sparse_weight_bounds'] = [-10.0, 10.0]
    sparse_config['nodeid_slot'] = nodeid_slot
    sparse_config['feature_learning_rate'] = feature_lr
    sparse_config['sparse_load_filter_slots'] = load_filter_slots
    return sparse_config


def generate_config(args):
    """ Generate Proto For PSlib  """
    config = dict()
    config['use_cvm'] = True
    config['trainer'] = "PSGPUTrainer"
    config['worker_class'] = "PSGPUWorker"
    config['use_ps_gpu'] = True
    # embedding name as key name
    # Id Embedding
    gen_config = gen_sparse_config

    slot_feature_lr = args.sparse_lr
    if "slot_feature_lr" in args:
        slot_feature_lr = args.slot_feature_lr
    if "train_storage_mode" in args and args.train_storage_mode == "SSD_EMBEDDING":
        sparse_table_class = "DownpourSparseSSDTable"
    else:
        sparse_table_class = "DownpourSparseTable"
    config['embedding'] = gen_config(args, args.sparse_lr, args.init_range, args.sparse_type, \
                                     args.emb_size, slot_feature_lr, args.nodeid_slot,
                                     args.load_filter_slots, sparse_table_class)

    dense_config = dict()
    dense_config['dense_table_class'] = "DownpourDenseTable"
    dense_config['dense_compress_in_save'] = True
    dense_config['dense_accessor_class'] = "DownpourDenseValueAccessor"
    dense_config['dense_learning_rate'] = args.dense_lr
    dense_config['dense_optimizer'] = "adam"
    dense_config['dense_avg_decay'] = 0.999993
    dense_config['dense_ada_decay'] = 0.9999
    dense_config['dense_ada_epsilon'] = 1e-8
    dense_config['dense_mom_decay'] = 0.99
    dense_config['dense_naive_lr'] = 0.0002
    # 'dense_table' as key name
    config['dense_table'] = dense_config

    datanorm_config = dict()
    datanorm_config['datanorm_table_class'] = "DownpourDenseTable"
    datanorm_config['datanorm_compress_in_save'] = True
    datanorm_config['datanorm_accessor_class'] = "DownpourDenseValueAccessor"
    datanorm_config['datanorm_operation'] = "summary"
    datanorm_config['datanorm_decay_rate'] = 0.999999
    config['datanorm_table'] = datanorm_config

    return config


class GraphRoleMaker(PaddleCloudRoleMaker):
    """ graph role maker """
    def __init__(self, is_collective=False, **kwargs):
        """ init """
        self._init_gloo = (paddle.distributed.get_world_size() > 1)
        super().__init__(
            is_collective=is_collective, init_gloo=self._init_gloo, **kwargs
        )
        self.gpu_nums = os.getenv("FLAGS_selected_gpus", 
                                  "0,1,2,3,4,5,6,7").split(",")

    def _init_collective_env(self):
        """ init gpu env """
        self._current_id = int(os.getenv("PADDLE_TRAINER_ID", "0"))
        self._training_role = os.getenv("PADDLE_TRAINING_ROLE", "TRAINER")
        assert self._training_role == "TRAINER"
        self._role = Role.WORKER
        self._worker_endpoints = os.getenv("PADDLE_TRAINER_ENDPOINTS")
        if self._worker_endpoints is None or len(self._worker_endpoints.split(":")) < 2:
            # back to non_distributed execution.
            self._worker_endpoints = "127.0.0.1:6170"
            self._cur_endpoint = self._worker_endpoints
            self._non_distributed = True
        self._worker_endpoints = self._worker_endpoints.split(",")
        self._cur_endpoint = self._worker_endpoints[paddle.distributed.get_rank()]
        self._nodes_num = paddle.distributed.get_world_size()
        self._trainers_num = self._nodes_num * len(self.gpu_nums)
        self._local_rank = paddle.distributed.get_rank() * len(self.gpu_nums)
        self._local_device_ids = self.gpu_nums
        self._world_device_ids = []
        self._non_distributed = (self._trainers_num == 1)
        
    def _worker_num(self):
        """ return trainer number """
        return self._trainers_num
    
    def _worker_index(self):
        """ return rank id """
        return self._local_rank
    
    def _role_id(self):
        """ return role id """
        return self._current_id
    
    def _node_num(self):
        """ node num """
        return self._nodes_num
    
    def _ps_endpoints(self):
        """ endpoints list """
        return self._worker_endpoints
    
    def _get_pserver_endpoints(self):
        """ pserver endpoints """
        return self._worker_endpoints
    
    def _generate_role(self):
        """
        generate role for role maker
        """
        if self._role_is_generated:
            return
        if not self._is_collective:
            super()._ps_env()
        else:
            self._init_collective_env()
        self._role_is_generated = True
        if not self._init_gloo:
            return
        self._gloo_init()
            
    def _gloo_init(self):
        # PADDLE_WITH_GLOO 1: trainer barrier, 2: all barrier
        use_gloo = int(os.getenv("PADDLE_WITH_GLOO", "0"))
        if use_gloo not in [1, 2]:
            return

        # PADDLE_GLOO_RENDEZVOUS 1: HDFS 2: FILE 3: HTTP
        rendezvous_type = int(os.getenv("PADDLE_GLOO_RENDEZVOUS", "0"))
        prefix = os.getenv("SYS_JOB_ID", "")
        if rendezvous_type not in [
            Gloo.RENDEZVOUS.HDFS,
            Gloo.RENDEZVOUS.HTTP,
            Gloo.RENDEZVOUS.FILE,
        ]:
            raise ValueError(self._gloo._err_type)

        need_init_all = True if use_gloo == 2 else False

        if rendezvous_type == Gloo.RENDEZVOUS.HDFS:
            dfs_name = os.getenv("PADDLE_GLOO_FS_NAME", "")
            dfs_ugi = os.getenv("PADDLE_GLOO_FS_UGI", "")
            dfs_path = os.getenv("PADDLE_GLOO_FS_PATH", "")
            kwargs = {
                "dfs.name": dfs_name,
                "dfs.ugi": dfs_ugi,
                "dfs.path": dfs_path,
                "store.prefix": prefix,
            }
        elif rendezvous_type == Gloo.RENDEZVOUS.HTTP:
            start_http_server = False
            manager = Manager()
            http_server_d = manager.dict()
            http_server_d["running"] = False
            if self._is_collective:
                ep_rank_0 = self._worker_endpoints[0]
                if self._is_first_worker():
                    start_http_server = True
            else:
                ep_rank_0 = os.getenv("PADDLE_GLOO_HTTP_ENDPOINT", "")
                if self._is_server() and self._server_index() == 0:
                    start_http_server = True
            ip, port = ep_rank_0.split(':')
            kwargs = {
                "http.host": ip,
                "http.port": port,
                "store.prefix": prefix,
                'start_http_server': start_http_server,
                'http_server_d': http_server_d,
            }
        else:
            dfs_path = os.getenv("PADDLE_GLOO_FS_PATH", "")
            kwargs = {
                "dfs.path": dfs_path,
                "store.prefix": prefix,
            }

        if rendezvous_type == Gloo.RENDEZVOUS.HDFS:
            type = "HDFS"
        elif rendezvous_type == Gloo.RENDEZVOUS.HTTP:
            type = "HTTP"
        else:
            type = "FILE"
        print(
            "Gloo init with {}: need_init_all: {}, args: {}".format(
                type, need_init_all, kwargs
            )
        )
        self._gloo.init(
            rendezvous=rendezvous_type,
            role=self._role,
            role_id=self._role_id(),
            worker_num=self._node_num(),
            server_num=self._server_num(),
            need_init_all=need_init_all,
            kwargs=kwargs,
        )

        if rendezvous_type == Gloo.RENDEZVOUS.HTTP:
            http_server_d['running'] = False
