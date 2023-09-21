#!/usr/bin/python
""" download graph data """
import os
import sys
import math
import time
import json
import subprocess
import multiprocessing
from multiprocessing import Process
from multiprocessing import Lock


def list_files(edge_path):
    """ list files """
    cmd = hadoop_fs + " -ls %s | awk '{print $8}'" % (edge_path)
    print("cmd %s" % cmd)
    pipe = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE).stdout
    lines = pipe.read()
    if (len(lines) == 0):
        print("list_files empty")
        exit(1)
    return lines.decode().strip().split('\n')


def init():
    """ init """
    sys.stdout.flush()
    os.system("mkdir %s > /dev/null 2>&1" % base_graph_data_path)
    all_edge_path = list_files(input_data_path)
    print("all edge path is %s" % all_edge_path)
    final_file_list = []
    for edge_path in all_edge_path:
        os.system("mkdir -p %s/%s > /dev/null 2>&1" %
                  (base_graph_data_path, edge_path.split('/')[-1]))
        hdfs_file_list = list_files(edge_path)
        final_file_list.extend(hdfs_file_list)
    with open('.download_locker', 'w') as f:
        json.dump(final_file_list, f)


def fini():
    """ fini operator """
    os.system('rm -rf .download_locker')
    print("download and shard model finished")
    sys.stdout.flush()


def excute_func(tid, lock):
    """ execute func """
    while (1):
        lock.acquire()
        hdfs_file_list = []
        with open('.download_locker') as f:
            hdfs_file_list = json.load(f)
        if (len(hdfs_file_list) == 0):
            lock.release()
            break
        hdfs_file = hdfs_file_list[0]
        del hdfs_file_list[0]
        with open('.download_locker', 'w') as f:
            json.dump(hdfs_file_list, f)
        lock.release()
        print("[%s][%d] download file[%s], remain %s..." % (time.strftime('%Y-%m-%d %H:%M:%S', \
            time.localtime(time.time())), tid, hdfs_file, len(hdfs_file_list)))
        hdfs_file_ls = hdfs_file.split('/')
        cmd = hadoop_fs + " -get %s %s/%s/%s" % (
            hdfs_file, base_graph_data_path, hdfs_file_ls[-2], hdfs_file_ls[-1]
        )
        if os.system(cmd) != 0:
            print("[%s][%d] exec cmd:[%s] fail, please check!" % (time.strftime('%Y-%m-%d %H:%M:%S', \
                time.localtime(time.time())), tid, cmd))
            exit(-1)
        print("[%s][%d] download file[%s] finished" % (time.strftime('%Y-%m-%d %H:%M:%S', \
            time.localtime(time.time())), tid, hdfs_file))
        sys.stdout.flush()


if __name__ == '__main__':
    input_data_path = sys.argv[1]
    fs_name = sys.argv[2]
    fs_ugi = sys.argv[3]
    hadoop_path = sys.argv[4]
    base_graph_data_path = sys.argv[5]
    hadoop_fs = "%s/bin/hadoop fs -Dfs.default.name=%s -Dhadoop.job.ugi=%s" % \
            (hadoop_path, fs_name, fs_ugi)
    print("input graph data path: %s" % input_data_path)
    cpu_num = multiprocessing.cpu_count()
    init()
    process_list = []
    lock = Lock()
    for i in range(0, cpu_num):
        p = Process(target=excute_func, args=(i, lock))
        p.start()
        process_list.append(p)

    for p in process_list:
        p.join()

    fini()
