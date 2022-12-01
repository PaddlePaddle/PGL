# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
"""Global Utilities Functions
"""
import os
import sys
import json
import time
import math
import shutil
import glob
import re
import traceback
import collections
import numpy as np
import pickle as pkl
from collections import defaultdict, OrderedDict
from datetime import datetime, timedelta

import paddle
paddle.enable_static()
import paddle.fluid as fluid
import paddle.fluid.core as core
import paddle.distributed.fleet as fleet

#from paddle.fluid.incubate.fleet.parameter_server.pslib import fleet
#from paddle.fluid.incubate.fleet.base.role_maker import GeneralRoleMaker
from pgl.utils.logger import log

import util_hadoop


def get_global_value(value_sum, value_cnt):
    """ get global value """
    value_sum = np.array(fluid.global_scope().find_var(value_sum.name).get_tensor())
    value_cnt = np.array(fluid.global_scope().find_var(value_cnt.name).get_tensor())
    return value_sum / np.maximum(value_cnt, 1)

def parse_path(path):
    """
    Args:
        path: path has follow format:
            1, /your/local/path
            2, afs:/your/remote/afs/path
            3, hdfs:/your/remote/hdfs/path

    Return:
        mode: 3 different modes: local, afs, hdfs
        output_path: /your/lcoal_or_remote/path

    """
    if path.startswith("afs"):
        mode = "afs"
        output_path = remove_prefix_of_hadoop_path(path)
    elif path.startswith("hdfs"):
        mode = "hdfs"
        output_path = remove_prefix_of_hadoop_path(path)
    else:
        mode = "local"
        output_path = path
    return mode, output_path

def remove_prefix_of_hadoop_path(hadoop_path):
    """
    Args:
        hadoop_path: afs://xxx.baidu.com:xxxx/your/remote/hadoop/path

    Return:
        output_path: /your/remote/hadoop/path
    """
    output_path = hadoop_path.split(":")[-1]
    output_path = re.split("^\d+", output_path)[-1]
    return output_path

def load_pretrained_model(exe, model_dict, args, model_path):
    """ load pretrained model """
    if fleet.is_first_worker():
        if os.path.exists(model_path): # local directory
            sparse_params_path = os.path.join(model_path, "000")
            if os.path.exists(sparse_params_path):
                log.info("[WARM] load sparse model from %s" % sparse_params_path)
                fleet.load_model(model_path, mode=0)
                log.info("[WARM] load sparse model from %s finished." %  sparse_params_path)
            else:
                log.info("[WARM] sparse model [%s] is not existed, skiped" % sparse_params_path)

            dense_params_path = os.path.join(model_path, "dense_vars")

        else:   # load from hadoop path
            mode, model_path = parse_path(model_path)
            model_path = util_hadoop.check_hdfs_path(model_path)
            sparse_params_path = os.path.join(model_path, "000")

            if util_hadoop.hdfs_exists(sparse_params_path):
                log.info("[WARM] load sparse model from %s" % sparse_params_path)
                fleet.load_model(model_path, mode=0)
                log.info("[WARM] load sparse model from %s finished." %  sparse_params_path)
            else:
                log.info("[WARM] sparse model [%s] is not existed, skipped" % sparse_params_path)

            hadoop_dense_params_path = os.path.join(model_path, "dense_vars")
            if util_hadoop.hdfs_exists(hadoop_dense_params_path):
                util_hadoop.hdfs_download(hadoop_dense_params_path, "./")
            dense_params_path = "./dense_vars"

        # load dense vars
        if os.path.exists(dense_params_path):
            log.info("[WARM] load dense parameters from: %s" % dense_params_path)
            all_vars = model_dict.train_program.global_block().vars
            for filename in os.listdir(dense_params_path):
                if filename in all_vars:
                    log.info("[WARM] var %s existed" % filename)
                else:
                    log.info("[WARM_MISS] var %s not existed" % filename)

            fluid.io.load_vars(exe,
                   dense_params_path,
                   model_dict.train_program,
                   predicate=name_not_have_sparse)
        elif args.pretrained_model: 
            # if hadoop model path did not include dense params, then load dense pretrained_model from dependency
            dependency_path = os.getenv("DEPENDENCY_HOME") # see env_run/scripts/train.sh for details
            dense_path = os.path.join(dependency_path, args.pretrained_model)
            log.info("[WARM] load dense parameters from: %s" % dense_path)
            paddle.static.set_program_state(model_dict.train_program, model_dict.state_dict)
        else:
            log.info("[WARM] dense model is not existed, skipped")

def save_pretrained_model(exe, save_path, mode="hdfs"):
    """save pretrained model"""
    if fleet.is_first_worker():
        if mode == "hdfs":
            save_path = util_hadoop.check_hdfs_path(save_path)
            util_hadoop.hdfs_rm(save_path)

        fleet.save_persistables(exe, save_path)

def name_not_have_sparse(var):
    """
    persistable var which not contains pull_box_sparse
    """
    res = "sparse" not in var.name and \
            fluid.io.is_persistable(var) and \
            var.name != "embedding" and \
            "learning_rate" not in var.name and \
            "_generated_var" not in var.name
    return res

def save_model(exe, model_dict, args, local_model_path, model_save_path):
    """final save model"""
    mode, model_save_path = parse_path(model_save_path)
    _, working_root = parse_path(args.working_root)

    # save sparse table
    log.info("save sparse table")
    if mode == "hdfs":
        save_pretrained_model(exe, model_save_path, mode = "hdfs")
    elif mode == "afs":
        save_pretrained_model(exe, local_model_path, mode = "local")
        user, passwd = args.fs_ugi.split(',')
        log.info("being to upload model to: %s " % model_save_path)
        gzshell_upload(args.fs_name, user, passwd, local_model_path, "afs:%s" % working_root)
        log.info("model has been saved, model_path: %s" % model_save_path)
    else:
        save_pretrained_model(exe, local_model_path, mode = "local")
        run_cmd("rm -rf %s && mkdir -p %s && mv %s %s" % \
                    (working_root, working_root, local_model_path, working_root))
        log.info("model has been saved in local path: %s" % working_root)

    # save dense model
    log.info("save dense model")
    local_var_save_path = "./dense_vars"
    if os.path.exists(local_var_save_path):
        shutil.rmtree(local_var_save_path)

    fluid.io.save_vars(exe,
            local_var_save_path,
            model_dict.train_program,
            predicate=name_not_have_sparse)

     # local_var_save_path is not existed if no variable in model
    if os.path.exists(local_var_save_path):
        if mode == "hdfs" or mode == "afs":
            util_hadoop.hdfs_upload(local_var_save_path, model_save_path)
        else:
            run_cmd("mv %s %s" % (local_var_save_path, model_save_path))

def upload_embedding(args, local_embed_path):
    mode, infer_result_path = parse_path(args.infer_result_path)
    _, working_root = parse_path(args.working_root)
    if mode == "hdfs":
        util_hadoop.hdfs_rm(infer_result_path)
        util_hadoop.hdfs_mkdir(infer_result_path)

        log.info("being to upload embedding to: %s " % infer_result_path)
        for file in glob.glob(os.path.join(local_embed_path, "*")):
            basename = os.path.basename(file)
            util_hadoop.hdfs_upload(file, infer_result_path)
        log.info("[hadoop put] embedding has been upload to: %s " % infer_result_path)

    elif mode == "afs":
        log.info("being to upload embedding to: %s " % infer_result_path)
        #  util_hadoop.hdfs_rm(infer_result_path)
        user, passwd = args.fs_ugi.split(',')
        gzshell_upload(args.fs_name, user, passwd, local_embed_path, "afs:%s" % working_root)
        log.info("[gzshell] embedding has been upload to: %s " % infer_result_path)
    else:
        make_dir(working_root)
        run_cmd("mv %s %s" % (local_embed_path, working_root))
        log.info("embedding has been saved in local path: %s" % working_root)

def hadoop_touch_done(path):
    """ touch hadoop done """
    if fleet.worker_index() == 0:
        with open("to.hadoop.done", 'w') as f:
            f.write("infer done\n")
        util_hadoop.hdfs_upload("to.hadoop.done", os.path.join(path, "to.hadoop.done"))

def print_useful_info():
    """ print useful info """
    try:
        import socket
        ip_addres = socket.gethostbyname(socket.gethostname())
        log.info("The IP_ADDRESS of this machine is: %s" % ip_addres)
    except Exception as e:
        log.info("%s" % (e))
        log.info("can not import socket")


# Global error handler
def global_except_hook(exctype, value, traceback):
    """global except hook"""
    import sys
    try:
        import mpi4py.MPI
        sys.stderr.write("\n*****************************************************\n")
        sys.stderr.write("Uncaught exception was detected on rank {}. \n".format(
            mpi4py.MPI.COMM_WORLD.Get_rank()))
        from traceback import print_exception
        print_exception(exctype, value, traceback)
        sys.stderr.write("*****************************************************\n\n\n")
        sys.stderr.write("\n")
        sys.stderr.write("Calling MPI_Abort() to shut down MPI processes...\n")
        sys.stderr.flush()
    finally:
        try:
            import mpi4py.MPI
            mpi4py.MPI.COMM_WORLD.Abort(1)
        except Exception as e:
            sys.stderr.write("*****************************************************\n")
            sys.stderr.write("Sorry, we failed to stop MPI, this process will hang.\n")
            sys.stderr.write("*****************************************************\n")
            sys.stderr.flush()
            raise e


def make_dir(path):
    """Build directory"""
    if not os.path.exists(path):
        os.makedirs(path)

def get_all_edge_type(etype2files, symmetry):
    """ get_all_edge_type """
    if symmetry:
        etype_list = []
        for etype in etype2files.keys():
            r_etype = get_inverse_etype(etype)
            etype_list.append(etype)
            if r_etype != etype:
                etype_list.append(r_etype)
    else:
        etype_list = list(etype2files.keys())

    return etype_list

def get_edge_type(etype, symmetry):
    """ get edge type with etype  """
    if symmetry:
        etype_list = []
        ori_type_list = etype.split(',')
        for i in range(0, len(ori_type_list)):
            etype_list.append(ori_type_list[i])
            r_etype = get_inverse_etype(ori_type_list[i])
            if r_etype != ori_type_list[i]:
                etype_list.append(r_etype)
    else:
        etype_list = etype.split(',')
    return etype_list

def get_inverse_etype(etype):
    """ get_inverse_etype """
    fields = etype.split("2")
    if len(fields) == 3:
        src, etype, dst = fields
        r_etype = "2".join([dst, etype, src])
    else:
        r_etype = "2".join([fields[1], fields[0]])
    return r_etype

def get_first_node_type(meta_path):
    """ get first node type from meta path """
    first_node = []
    meta_paths = meta_path.split(';')
    for i in range(len(meta_paths)):
        tmp_node = meta_paths[i].split('2')[0]
        first_node.append(tmp_node)
    return ";".join(first_node)

def parse_files(type_files):
    """ parse_files """
    type2files = OrderedDict()
    for item in type_files.split(","):
        t, file_or_dir = item.split(":")
        type2files[t] = file_or_dir
    return type2files


def get_files(edge_file_or_dir):
    """ get_files """
    if os.path.isdir(edge_file_or_dir):
        ret_files = []
        files = sorted(glob.glob(os.path.join(edge_file_or_dir, "*")))
        for file_ in files:
            if os.path.isdir(file_):
                log.info("%s is a directory, not a file" % file_)
            else:
                ret_files.append(file_)
    elif "*" in edge_file_or_dir:
        ret_files = []
        files = glob.glob(edge_file_or_dir)
        for file_ in files:
            if os.path.isdir(file_):
                log.info("%s is a directory, not a file" % file_)
            else:
                ret_files.append(file_)
    else:
        ret_files = [edge_file_or_dir]
    return ret_files


def load_ip_addr(ip_config):
    """ load_ip_addr """
    if isinstance(ip_config, str):
        ip_addr_list = []
        with open(ip_config, 'r') as f:
            for line in f:
                ip_addr_list.append(line.strip())
        ip_addr = ";".join(ip_addr_list)
    elif isinstance(ip_config, list):
        ip_addr = ";".join(ip_config)
    else:
        raise TypeError("ip_config should be list of IP address or "
                        "a path of IP configuration file. "
                        "But got %s" % (type(ip_config)))
    return ip_addr


def convert_nfeat_info(nfeat_info):
    """ convert_nfeat_info """
    res = defaultdict(dict)
    for item in nfeat_info:
        res[item[0]].update({item[1]: [item[2], item[3]]})
    return res

def gzshell_upload(fs_name, fs_user, fs_password, local_path, remote_path):
    """ upload data with gzshell in afs """
    gzshell = os.getenv("GZSHELL")
    client_conf = os.getenv("CLIENT_CONF")
    cmd = "%s --uri=%s --username=%s --password=%s --conf=%s \
            --thread=100 -put %s/ %s" % (gzshell, fs_name, fs_user, \
            fs_password, client_conf, local_path, remote_path)
    upload_res = run_cmd_get_return_code(cmd)
    retry_num = 0
    while upload_res != 0:
        if retry_num > 3:
            log.info("upload model failed exceeds retry num limit!")
            break
        upload_res = run_cmd_get_return_code(cmd)
        retry_num += 1
    if upload_res != 0:
        log.info("save afs model failed!")
        exit(-1)

def run_cmd_get_return_code(cmd):
    """
    run cmd and get its return code, 0 means correct
    """
    return int(core.run_cmd(cmd + "; echo $?").strip().split('\n')[-1])

def run_cmd(cmd):
    """
    run cmd and check result
    """
    ret = run_cmd_get_return_code(cmd)
    if ret != 0:
        raise RuntimeError("Fail to run cmd[%s] ret[%d]" % (cmd, ret))
    return ret
