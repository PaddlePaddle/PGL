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
"""Hadoop Utilities Functions
"""
import os
import sys
import json
import time
import math
import collections
import numpy as np

from util_config import prepare_config
curPath = os.path.abspath(os.path.dirname(__file__))
sys.path.append(curPath)
config = prepare_config(os.path.join(curPath, "config.yaml"))

Ddfs = " -Ddfs.client.block.write.retries=15 -Ddfs.rpc.timeout=300000 -Ddfs.delete.trash=1"

def check_hdfs_path(path, fs_name=config.fs_name, fs_ugi=config.fs_ugi):
    """check hdfs path"""
    if path.startswith("hdfs://") or path.startswith("afs://"):
        return path
    else:
        real_path = fs_name + path
        return real_path

def hdfs_ls(path, fs_name=config.fs_name, fs_ugi=config.fs_ugi):
    """ hdfs_ls """
    path = check_hdfs_path(path, fs_name, fs_ugi)
    cmd = "hadoop fs -D fs.default.name=" + fs_name
    cmd += " -D hadoop.job.ugi=" + fs_ugi
    cmd += Ddfs
    cmd += " -ls " + path
    cmd += " | grep part | awk '{print $8}'"
    filelist = os.popen(cmd).read().split()
    return filelist


def hdfs_exists(path, fs_name=config.fs_name, fs_ugi=config.fs_ugi):
    """ hdfs_exists """
    path = check_hdfs_path(path, fs_name, fs_ugi)
    cmd = "hadoop fs -D fs.default.name=" + fs_name
    cmd += " -D hadoop.job.ugi=" + fs_ugi
    cmd += Ddfs
    cmd += " -test -e " + path + " 2>/dev/null ; echo $?"
    ret = int(os.popen(cmd).read().strip())
    if ret == 0:
        return True
    return False


def hdfs_replace(src, dest, fs_name=config.fs_name, fs_ugi=config.fs_ugi):
    """ hdfs_replace """
    src = check_hdfs_path(src, fs_name, fs_ugi)
    dest = check_hdfs_path(dest, fs_name, fs_ugi)

    tmp = dest + "_" + str(int(time.time()))
    cmd = "hadoop fs -D fs.default.name=" + fs_name
    cmd += " -D hadoop.job.ugi=" + fs_ugi
    cmd += Ddfs
    cmd += " -mv " + dest + " " + tmp + " && "

    cmd += " hadoop fs -D fs.default.name=" + fs_name
    cmd += " -D hadoop.job.ugi=" + fs_ugi
    cmd += Ddfs
    cmd += " -put " + src + " " + dest + " && "

    cmd += " hadoop fs -D fs.default.name=" + fs_name
    cmd += " -D hadoop.job.ugi=" + fs_ugi
    cmd += Ddfs
    cmd += " -rmr " + tmp
    ret = os.system(cmd)
    return ret

def hdfs_mv(src, dest, fs_name=config.fs_name, fs_ugi=config.fs_ugi):
    """ hdfs_replace """
    src = check_hdfs_path(src, fs_name, fs_ugi)
    dest = check_hdfs_path(dest, fs_name, fs_ugi)

    if hdfs_exists(dest):
        hdfs_rm(dest)
    cmd = "hadoop fs -D fs.default.name=" + fs_name
    cmd += " -D hadoop.job.ugi=" + fs_ugi
    cmd += Ddfs
    cmd += " -mv " + src + " " + dest

    ret = os.system(cmd)
    return ret

def hdfs_rm(path, fs_name=config.fs_name, fs_ugi=config.fs_ugi):
    """ hdfs_rm """
    path = check_hdfs_path(path, fs_name, fs_ugi)
    cmd = "hadoop fs -D fs.default.name=" + fs_name
    cmd += " -D hadoop.job.ugi=" + fs_ugi
    cmd += Ddfs
    cmd += " -rmr " + path

    if hdfs_exists(path):
        ret = os.system(cmd)
        return ret
    else:
        return 0

def hdfs_append(filename, dest, fs_name=config.fs_name, fs_ugi=config.fs_ugi):
    """ hdfs_append """
    cmd = "hadoop fs -D fs.default.name=" + fs_name
    cmd += " -D hadoop.job.ugi=" + fs_ugi
    cmd += Ddfs
    cmd += " -appendToFile " + filename + " " + dest
    ret = os.system(cmd)
    return ret


def hdfs_cat(filename, fs_name=config.fs_name, fs_ugi=config.fs_ugi):
    """ hdfs_cat """
    filename = check_hdfs_path(filename, fs_name, fs_ugi)
    cmd = "hadoop fs -D fs.default.name=" + fs_name
    cmd += " -D hadoop.job.ugi=" + fs_ugi
    cmd += Ddfs
    cmd += " -cat " + filename
    p = os.popen(cmd)
    return p.read().strip()


def hdfs_file_open(filename, fs_name=config.fs_name, fs_ugi=config.fs_ugi):
    """ hdfs_file_open """
    filename = check_hdfs_path(filename, fs_name, fs_ugi)
    cmd = "hadoop fs -D fs.default.name=" + fs_name
    cmd += " -D hadoop.job.ugi=" + fs_ugi
    cmd += Ddfs
    cmd += " -cat " + filename
    p = os.popen(cmd)
    return p

def hdfs_gz_file_open(filename, fs_name=config.fs_name, fs_ugi=config.fs_ugi):
    """ hdfs_file_open """
    filename = check_hdfs_path(filename, fs_name, fs_ugi)
    cmd = "hadoop fs -D fs.default.name=" + fs_name
    cmd += " -D hadoop.job.ugi=" + fs_ugi
    cmd += " -cat " + filename + " | zcat"
    p = os.popen(cmd)
    return p



def hdfs_download(src, dest, fs_name=config.fs_name, fs_ugi=config.fs_ugi):
    """ hdfs_download """
    cmd = "hadoop fs -D fs.default.name=" + fs_name
    cmd += " -D hadoop.job.ugi=" + fs_ugi
    cmd += Ddfs
    cmd += " -get " + src + " " + dest
    ret = os.system(cmd)
    return ret


def hdfs_upload(src, dest, fs_name=config.fs_name, fs_ugi=config.fs_ugi):
    """ hdfs_upload """
    cmd = "hadoop fs -D fs.default.name=" + fs_name
    cmd += " -D hadoop.job.ugi=" + fs_ugi
    cmd += Ddfs
    cmd += " -put " + src + " " + dest
    ret = os.system(cmd)
    return ret


def hdfs_mkdir(path, fs_name=config.fs_name, fs_ugi=config.fs_ugi):
    """ hdfs_mkdir """
    path = check_hdfs_path(path, fs_name, fs_ugi)
    # mkdir
    cmd = "hadoop fs -D fs.default.name=" + fs_name
    cmd += " -D hadoop.job.ugi=" + fs_ugi
    cmd += Ddfs
    cmd += " -mkdir " + path
    ret = os.system(cmd)
    return ret


