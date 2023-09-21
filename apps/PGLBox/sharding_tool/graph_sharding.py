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
"""Graph Sharding for PGLBox
"""

import sys
import os
import logging
import time
import json
import shutil
import argparse
import multiprocessing
from multiprocessing import Lock
from multiprocessing import Process

LOG_FILE = "shard.log"
logger = logging.getLogger('graph_sharding')
logger.setLevel(logging.DEBUG)
fmt = logging.Formatter('%(levelname)s: %(asctime)s %(process)d'
                        ' [%(filename)s:%(lineno)s][%(funcName)s] %(message)s')
debug_handler = logging.FileHandler(LOG_FILE, 'a')
debug_handler.setFormatter(fmt)
debug_handler.setLevel(logging.DEBUG)
logger.addHandler(debug_handler)
print("the sharding log will be saved in %s" % LOG_FILE)

TEMP_FILE = ".tmp_sharding_file"
CPU_NUM = multiprocessing.cpu_count()


def mapper(args, input_file, lock_list, proc_index):
    """
    mapper
    """
    logger.debug('processing %s' % input_file)
    start_time = time.time()
    cache = []
    for i in range(args.part_num):
        cache.append([])

    with open(input_file, "r") as reader:
        for line in reader:
            line = line.rstrip("\r\n")
            fields = line.split("\t")
            if args.node_type_shard:  # node type shard
                nid = int(fields[1])
                cache[nid % args.part_num].append(line)
            else:  # edge shard
                src_nid = int(fields[0])
                cache[src_nid % args.part_num].append(line)

                if args.symmetry:
                    dst_nid = int(fields[1])
                    if src_nid != dst_nid:
                        cache[dst_nid % args.part_num].append(line)

    read_time = time.time()

    for i in range(proc_index, args.part_num + proc_index):
        index = i % args.part_num
        lock_list[index].acquire()
        with open(os.path.join(args.output_dir, "part-%05d" % index),
                  'a+') as writer:
            for item in cache[index]:
                writer.write("%s\n" % item)
        lock_list[index].release()

    output_time = time.time()
    logger.debug('processed %s, read_time[%f] write_time[%f]' % (
        input_file, (read_time - start_time), (output_time - read_time)))


class FileMapper(object):
    """
    FileMapper
    """

    def init(self, input_dir):
        """ Init """
        file_list = os.listdir(input_dir)
        new_file_list = []
        for file in file_list:
            new_file_list.append(input_dir + '/' + file)
        file_list = new_file_list

        with open(TEMP_FILE, 'w') as f:
            json.dump(file_list, f)

        return len(file_list)

    def fini(self):
        """ Fini """
        os.system('rm -rf %s' % TEMP_FILE)
        sys.stdout.flush()

    def excute_func(self, args, func, progress_file_lock, lock_list,
                    proc_index):
        """ Repeat download one file """
        while (1):
            progress_file_lock.acquire()
            file_list = []

            with open(TEMP_FILE) as f:
                file_list = json.load(f)
            if (len(file_list) == 0):
                progress_file_lock.release()
                break

            input_file = file_list[0]
            del file_list[0]
            with open(TEMP_FILE, 'w') as f:
                json.dump(file_list, f)
            progress_file_lock.release()

            func(args, input_file, lock_list, proc_index)

    def run(self, args, func):
        """ Run """
        logger.debug('begin Run')
        file_num = self.init(args.input_dir)
        process_list = []
        lock_list = []
        progress_file_lock = Lock()
        for i in range(args.part_num):
            lock_list.append(Lock())
        process_num = min(args.num_workers, file_num, CPU_NUM)
        for i in range(0, process_num):
            p = Process(
                target=self.excute_func,
                args=(args, func, progress_file_lock, lock_list, i))
            p.start()
            process_list.append(p)

        for p in process_list:
            p.join()
        self.fini()
        logger.debug('end Run')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='graph_sharding')
    parser.add_argument("--input_dir", type=str, default="/pglbox/raw_data")
    parser.add_argument("--output_dir", type=str, default="/pglbox/new_data")
    parser.add_argument("--part_num", type=int, default=1000)
    parser.add_argument("--num_workers", type=int, default=200)
    parser.add_argument("--node_type_shard", action='store_true')
    parser.add_argument("--symmetry", action='store_true')
    args = parser.parse_args()
    logger.debug(args)

    if os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)
    os.makedirs(args.output_dir)

    file_mapper = FileMapper()
    file_mapper.run(args, mapper)
