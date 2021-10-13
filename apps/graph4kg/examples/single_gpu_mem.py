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
"""
统计显卡利用率
"""
# encoding: utf-8
import os
import sys
import time
import subprocess
import logging

import cup
import py3nvml  # only for python3.7
import numpy as np

FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)


def get_gpu_mem(gpu_id=0):
    """
    get gpu mem from gpu id
    Args:
        gpu_id (int): gpu id
    """
    py3nvml.py3nvml.nvmlInit()
    gpu_handle = py3nvml.py3nvml.nvmlDeviceGetHandleByIndex(gpu_id)
    gpu_mem_info = py3nvml.py3nvml.nvmlDeviceGetMemoryInfo(gpu_handle)
    gpu_utilization_info = py3nvml.py3nvml.nvmlDeviceGetUtilizationRates(
        gpu_handle)
    gpu_mem = {}
    gpu_mem['total(MB)'] = gpu_mem_info.total / 1024.**2
    gpu_mem['free(MB)'] = gpu_mem_info.free / 1024.**2
    gpu_mem['used(MB)'] = gpu_mem_info.used / 1024.**2
    gpu_mem['gpu_utilization_rate(%)'] = gpu_utilization_info.gpu
    gpu_mem['gpu_mem_utilization_rate(%)'] = gpu_utilization_info.memory
    py3nvml.py3nvml.nvmlShutdown()
    return gpu_mem


def get_pid(name):
    """
    get pid from process name
    """
    try:
        pid = 0
        command = "ps aux |grep '{name}' | tr -s ' '| cut -d ' ' -f 2".format(
            name=name)
        p = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
        out, err = p.communicate()
        for line in out.splitlines():
            pid = int(line)
            break
        print(pid)
    except subprocess.CalledProcessError:
        logger.warning(subprocess.CalledProcessError)
        logger.error("No pid was detected, record process will end")
        pid = None
    return pid


def check_pid_status(pid):
    """
    根据pid来监测进程状态，放到循环里去
    """
    if pid:
        # 判断进程目录是否存在
        import os
        dirs = "/proc/{pid}".format(pid=pid)
        return os.path.exists(dirs)
    return False


def get_cpu_mem(pid):
    """
    get cpu mem from pid
    """
    if pid is None:
        logger.warning("process pid is None, end process")
        return None
    process = cup.res.Process(pid)
    mem_info = process.get_ext_memory_info()
    mem = {}
    mem['process_name'] = process.get_process_name()
    mem['rss(MB)'] = mem_info.rss / 1024.**2
    mem['vms(MB)'] = mem_info.vms / 1024.**2
    mem['shared(MB)'] = mem_info.shared / 1024.**2
    mem['dirty(MB)'] = mem_info.dirty / 1024.**2
    mem['cpu_usage(%)'] = cup.res.linux.get_cpu_usage(intvl_in_sec=1).usr
    return mem


def summary(cpu_mem_lists, gpu_mem_lists=None):
    """
    return reports of cpu and gpu info
    """
    cpu_reports = {}
    gpu_reports = {}

    cpu_reports['process_name'] = cpu_mem_lists[0]['process_name']
    cpu_reports['rss(MB)'] = max([float(i['rss(MB)']) for i in cpu_mem_lists])
    cpu_reports['vms(MB)'] = max([float(i['vms(MB)']) for i in cpu_mem_lists])
    cpu_reports['shared(MB)'] = max(
        [float(i['shared(MB)']) for i in cpu_mem_lists])
    cpu_reports['dirty(MB)'] = max(
        [float(i['dirty(MB)']) for i in cpu_mem_lists])
    cpu_reports['cpu_usage(%)'] = max(
        [float(i['cpu_usage(%)']) for i in cpu_mem_lists])

    cpu_reports['rss(MB)_mean'] = np.mean(
        [float(i['rss(MB)']) for i in cpu_mem_lists])
    cpu_reports['vms(MB)_mean'] = np.mean(
        [float(i['vms(MB)']) for i in cpu_mem_lists])
    cpu_reports['shared(MB)_mean'] = np.mean(
        [float(i['shared(MB)']) for i in cpu_mem_lists])
    cpu_reports['dirty(MB)_mean'] = np.mean(
        [float(i['dirty(MB)']) for i in cpu_mem_lists])
    cpu_reports['cpu_usage(%)_mean'] = np.mean(
        [float(i['cpu_usage(%)']) for i in cpu_mem_lists])

    logger.info("----------------------- Res info -----------------------")
    logger.info("process_name: {0}, cpu rss(MB): {1} (mean: {2}), \
vms(MB): {3} (mean {4}), shared(MB): {5} (mean {6}), dirty(MB): {7} (mean {8}), \
cpu_usage(%): {9} (mean {10}) ".format(cpu_reports[
        'process_name'], cpu_reports['rss(MB)'], cpu_reports[
            'rss(MB)_mean'], cpu_reports['vms(MB)'], cpu_reports[
                'vms(MB)_mean'], cpu_reports['shared(MB)'], cpu_reports[
                    'shared(MB)_mean'], cpu_reports['dirty(MB)'], cpu_reports[
                        'dirty(MB)_mean'], cpu_reports['cpu_usage(%)'],
                                       cpu_reports['cpu_usage(%)_mean']))

    if gpu_mem_lists:
        logger.info("=== gpu info was recorded ===")
        gpu_reports['gpu_id'] = int(os.environ.get("CUDA_VISIBLE_DEVICES"))
        gpu_reports['total(MB)'] = max(
            [float(i['total(MB)']) for i in gpu_mem_lists])
        gpu_reports['free(MB)'] = max(
            [float(i['free(MB)']) for i in gpu_mem_lists])
        gpu_reports['used(MB)'] = max(
            [float(i['used(MB)']) for i in gpu_mem_lists])
        gpu_reports['gpu_utilization_rate(%)'] = max(
            [float(i['gpu_utilization_rate(%)']) for i in gpu_mem_lists])
        gpu_reports['gpu_mem_utilization_rate(%)'] = max(
            [float(i['gpu_mem_utilization_rate(%)']) for i in gpu_mem_lists])

        gpu_reports['total(MB)_mean'] = np.mean(
            [float(i['total(MB)']) for i in gpu_mem_lists])
        gpu_reports['free(MB)_mean'] = np.mean(
            [float(i['free(MB)']) for i in gpu_mem_lists])
        gpu_reports['used(MB)_mean'] = np.mean(
            [float(i['used(MB)']) for i in gpu_mem_lists])
        gpu_reports['gpu_utilization_rate(%)_mean'] = np.mean(
            [float(i['gpu_utilization_rate(%)']) for i in gpu_mem_lists])
        gpu_reports['gpu_mem_utilization_rate(%)_mean'] = np.mean(
            [float(i['gpu_mem_utilization_rate(%)']) for i in gpu_mem_lists])

        logger.info("gpu_id: {0}, total(MB): {1} (mean {2}), \
free(MB): {3} (mean {4}), used(MB): {5} (mean {6}), gpu_utilization_rate(%): {7} (mean {8}), \
gpu_mem_utilization_rate(%): {9} (mean {10}) ".format(
            gpu_reports['gpu_id'], gpu_reports['total(MB)'], gpu_reports[
                'total(MB)_mean'], gpu_reports['free(MB)'],
            gpu_reports['free(MB)_mean'], gpu_reports['used(MB)'], gpu_reports[
                'used(MB)_mean'], gpu_reports['gpu_utilization_rate(%)'],
            gpu_reports['gpu_utilization_rate(%)_mean'], gpu_reports[
                'gpu_mem_utilization_rate(%)'], gpu_reports[
                    'gpu_mem_utilization_rate(%)_mean']))
        return cpu_reports, gpu_reports
    else:
        return cpu_reports, None


def main():
    """
    main
    """
    process = sys.argv[1]
    step = float(sys.argv[2])
    time.sleep(0.5)
    use_gpu = True
    s_time = time.time()
    gpu_ids = os.environ.get("CUDA_VISIBLE_DEVICES")  # 0,1,2,3
    if gpu_ids is not None:
        gpu_id_lists = gpu_ids.split(',')
        if len(gpu_id_lists) > 1:
            logger.warning("more than one CUDA_VISIBLE_DEVICES was set")
        else:
            gpu_id = int(gpu_ids)
            logger.info('training on gpu %d' % gpu_id)
    else:
        use_gpu = False
    cpu_mem_lists = []
    gpu_mem_lists = []
    pid = get_pid(process)
    while check_pid_status(pid):
        if pid:
            cpu_mem = get_cpu_mem(pid)
            cpu_mem_lists.append(cpu_mem)
            if use_gpu:
                gpu_mem = get_gpu_mem(gpu_id)
                gpu_mem_lists.append(gpu_mem)
        else:
            logger.warning("==== process pid is None, end recording ===")
            break
        time.sleep(5)
        if time.time() - s_time > step:
            summary(cpu_mem_lists, gpu_mem_lists)
            s_time = time.time()

    # print(cpu_reports)
    # print(gpu_reports)


if __name__ == "__main__":
    main()
