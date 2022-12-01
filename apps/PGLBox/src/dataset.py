import os
import time
import paddle
import threading
from paddle.distributed import fleet
import util
import paddle.fluid as fluid
from place import get_cuda_places
from pgl.utils.logger import log


class BaseDataset(object):
    def __init__(self, chunk_num, dataset_config, holder_list, embedding=None, graph=None):
        self.ins_ready_sem = threading.Semaphore(0)
        self.could_load_sem = threading.Semaphore(2)
        self.graph = graph
        self.embedding = embedding
        self.chunk_num = chunk_num
        self.holder_list = holder_list

    def generate_dataset(self, chunk_index):
        raise NotImplementedError 

    def chunk_generator(self):
        # open a thread for processing the data
        dataset_list = []
        t = threading.Thread(target=self.preload_thread, args=(dataset_list, ))
        t.setDaemon(True)
        t.start()

        for chunk_index in range(self.chunk_num):
            index = fleet.worker_index() * self.chunk_num + chunk_index
            global_chunk_num = fleet.worker_num() * self.chunk_num
            self.ins_ready_sem.acquire()
            dataset = dataset_list[chunk_index]

            if dataset is None:
                self.could_load_sem.release()
                continue

            beginpass_begin = time.time()
            self.embedding.begin_pass()
            beginpass_end = time.time()
            log.info("STAGE [BEGIN PASS] finished, time cost: %f sec" \
                    % (beginpass_end - beginpass_begin))

            yield dataset
            endpass_begin = time.time()
            self.embedding.end_pass()
            endpass_end = time.time()
            log.info("STAGE [END PASS] finished, time cost: %f sec" \
                    % (endpass_end - endpass_begin))
   
            dataset.release_memory()
            self.could_load_sem.release()
        t.join()

    def get_file_list(self, chunk_index):
        """ get data file list """
        work_dir = "./workdir/filelist" # a tmp directory that does not used in other places

        if not os.path.isdir(work_dir):
            os.makedirs(work_dir)

        file_list = []
        for thread_id in range(len(get_cuda_places())):
            filename = os.path.join(work_dir, "%s_%s_%s" % (self.chunk_num, chunk_index, thread_id))
            file_list.append(filename)
            with open(filename, "w") as writer:
                writer.write("%s_%s_%s\n" % (self.chunk_num, chunk_index, thread_id))
        return file_list
    
    def preload_thread(self, dataset_list):
        """ This is a thread to fill the dataset_list
        """
        try:
            self.preload_worker(dataset_list)
        except Exception as e:
            self.could_load_sem.release()
            self.ins_ready_sem.release()
            log.info('preload_thread exception :%s' % (e))

    def preload_worker(self, dataset_list):
        """ This is a preload worker to generate chunk dataset asynchronously
        """
        for chunk_index in range(self.chunk_num):
            dataset = None
            self.could_load_sem.acquire()
            if dataset is not None:
                begin = time.time()
                dataset.wait_preload_done()
                end = time.time()
                log.info("wait data preload done cost %s min" % ((end - begin) / 60.0))

            if dataset is None:
                index = fleet.worker_index() * self.chunk_num + chunk_index
                global_chunk_num = fleet.worker_num() * self.chunk_num
                dataset = self.generate_dataset(index)
                log.info("going to load into memory")


                begin = time.time()
                dataset.load_into_memory(is_shuffle=False)
                end = time.time()
                log.info("load into memory + shuffle done, cost %s min" % ((end - begin) / 60.0))
                log.info("get_memory_data_size: %d" % (dataset.get_memory_data_size()))
                dataset_list.append(dataset)
                self.ins_ready_sem.release()
        log.info("thread finished, exit")


class UnsupReprLearningDataset(BaseDataset):
    def __init__(self, chunk_num, dataset_config, holder_list, embedding=None, graph=None):

        self.dataset_config = dataset_config

        self.walk_len = dataset_config.walk_len
        self.walk_times = dataset_config.walk_times
        self.batch_node_size = dataset_config.batch_node_size
        self.sample_times_one_chunk = dataset_config.sample_times_one_chunk
        self.win_size = dataset_config.win_size
        self.debug_mode = dataset_config.debug_mode
        self.batch_size = dataset_config.batch_size
        self.meta_path = dataset_config.meta_path
        self.sage_mode = dataset_config.sage_mode
        self.samples = dataset_config.samples
        self.fs_name =  dataset_config.fs_name if dataset_config.fs_name is not None else ""
        self.fs_ugi =  dataset_config.fs_ugi if dataset_config.fs_ugi is not None else ""

        super(UnsupReprLearningDataset, self).__init__(chunk_num=chunk_num,
                      dataset_config=dataset_config,
                      holder_list=holder_list,
                      embedding=embedding,
                      graph=graph)
    

    def generate_dataset(self, chunk_index):
        sage_mode = self.sage_mode if self.sage_mode else False

        str_samples = ""
        if sage_mode and self.samples and len(self.samples) > 0:
            for s in self.samples:
                str_samples += str(s)
                str_samples += ";"
            str_samples = str_samples[:-1]

        graph_config = {"walk_len": self.walk_len,
                    "walk_degree": self.walk_times,
                    "once_sample_startid_len": self.batch_node_size,
                    "sample_times_one_chunk": self.sample_times_one_chunk,
                    "window": self.win_size,
                    "debug_mode": self.debug_mode,
                    "batch_size": self.batch_size,
                    "meta_path": self.meta_path,
                    "gpu_graph_training": True,
                    "sage_mode": sage_mode,
                    "samples": str_samples}

        first_node_type = util.get_first_node_type(self.meta_path)
        graph_config["first_node_type"] = first_node_type

        dataset = fluid.DatasetFactory().create_dataset("InMemoryDataset")
        dataset.set_feed_type("SlotRecordInMemoryDataFeed")
        dataset.set_use_var(self.holder_list)
        dataset.set_graph_config(graph_config)

        dataset.set_batch_size(1)  # Fixed Don't Change. Batch Size is not config here.
        dataset.set_thread(len(get_cuda_places()))

        dataset.set_hdfs_config(self.fs_name, self.fs_ugi)

        dataset._set_use_ps_gpu(self.embedding.parameter_server)
        file_list = self.get_file_list(chunk_index) 
        dataset.set_filelist(file_list)
        with open("datafeed.pbtxt", "w") as fout:
            fout.write(dataset.desc())
        return dataset


class InferDataset(BaseDataset):
    def __init__(self, chunk_num, dataset_config, holder_list, embedding=None, graph=None):

        self.dataset_config = dataset_config

        self.walk_len = dataset_config.walk_len
        self.walk_times = dataset_config.walk_times
        self.batch_node_size = dataset_config.batch_node_size
        self.sample_times_one_chunk = dataset_config.sample_times_one_chunk
        self.win_size = dataset_config.win_size
        self.debug_mode = dataset_config.debug_mode
        self.batch_size = dataset_config.infer_batch_size
        self.meta_path = dataset_config.meta_path
        self.sage_mode = dataset_config.sage_mode
        self.samples = dataset_config.infer_samples
        self.fs_name =  dataset_config.fs_name if dataset_config.fs_name is not None else ""
        self.fs_ugi =  dataset_config.fs_ugi if dataset_config.fs_ugi is not None else ""

        super(InferDataset, self).__init__(chunk_num=chunk_num,
                      dataset_config=dataset_config,
                      holder_list=holder_list,
                      embedding=embedding,
                      graph=graph)
    

    def generate_dataset(self, chunk_index):
        sage_mode = self.sage_mode if self.sage_mode else False

        str_samples = ""
        if sage_mode and self.samples and len(self.samples) > 0:
            for s in self.samples:
                str_samples += str(s)
                str_samples += ";"
            str_samples = str_samples[:-1]

        graph_config = {"walk_len": self.walk_len,
                    "walk_degree": self.walk_times,
                    "once_sample_startid_len": self.batch_node_size,
                    "sample_times_one_chunk": self.sample_times_one_chunk,
                    "window": self.win_size,
                    "debug_mode": self.debug_mode,
                    "batch_size": self.batch_size,
                    "meta_path": self.meta_path,
                    "gpu_graph_training": False,
                    "sage_mode": sage_mode,
                    "samples": str_samples}

        first_node_type = util.get_first_node_type(self.meta_path)
        graph_config["first_node_type"] = first_node_type

        dataset = fluid.DatasetFactory().create_dataset("InMemoryDataset")
        dataset.set_feed_type("SlotRecordInMemoryDataFeed")
        dataset.set_use_var(self.holder_list)
        dataset.set_graph_config(graph_config)

        dataset.set_batch_size(1)  # Fixed Don't Change. Batch Size is not config here.
        dataset.set_thread(len(get_cuda_places()))

        dataset.set_hdfs_config(self.fs_name, self.fs_ugi)

        dataset._set_use_ps_gpu(self.embedding.parameter_server)
        file_list = self.get_file_list(chunk_index) 
        dataset.set_filelist(file_list)
        with open("datafeed.pbtxt", "w") as fout:
            fout.write(dataset.desc())
        return dataset


