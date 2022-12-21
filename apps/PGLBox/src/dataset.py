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
    def __init__(self, chunk_num, config, holder_list, embedding=None, graph=None, is_predict=False):
        self.ins_ready_sem = threading.Semaphore(0)
        self.could_load_sem = threading.Semaphore(2)
        self.graph = graph
        self.config = config
        self.embedding = embedding
        self.chunk_num = chunk_num
        self.holder_list = holder_list
        self.is_predict = is_predict

        if self.is_predict:
            if hasattr(self.embedding, "set_mode"):
                self.embedding.set_mode(True)

    def generate_dataset(self, config, chunk_index):
        sage_mode = config.sage_mode if config.sage_mode else False
        fs_name = config.fs_name if config.fs_name is not None else ""
        fs_ugi = config.fs_ugi if config.fs_ugi is not None else ""

        str_samples = ""
        if sage_mode and config.samples and len(config.samples) > 0:
            for s in config.samples:
                str_samples += str(s)
                str_samples += ";"
            str_samples = str_samples[:-1]

        str_infer_samples = ""
        if sage_mode and config.infer_samples and len(config.infer_samples) > 0:
            for s in config.infer_samples:
                str_infer_samples += str(s)
                str_infer_samples += ";"
            str_infer_samples = str_infer_samples[:-1]

        excluded_train_pair = config.excluded_train_pair if config.excluded_train_pair else ""
        infer_node_type = config.infer_node_type if config.infer_node_type else ""

        uniq_factor = 0.4
        if not sage_mode:
            train_pass_cap = int(config.walk_len * config.walk_times * config.sample_times_one_chunk \
                             * config.batch_node_size * uniq_factor)
        else:
            # If sage_mode is True, self.samples can not be None.
            train_pass_cap = int(config.walk_len * config.walk_times * config.sample_times_one_chunk \
                             * config.batch_node_size * uniq_factor * config.samples[0])

        infer_pass_cap = 10000000  # 1kw
        if config.train_pass_cap:
            train_pass_cap = config.train_pass_cap
        if config.infer_pass_cap:
            infer_pass_cap = config.infer_pass_cap

        get_degree = sage_mode and (config.use_degree_norm
                                    if config.use_degree_norm else False)

        graph_config = {
            "walk_len": config.walk_len,
            "walk_degree": config.walk_times,
            "once_sample_startid_len": config.batch_node_size,
            "sample_times_one_chunk": config.sample_times_one_chunk,
            "window": config.win_size,
            "debug_mode": config.debug_mode,
            "batch_size": config.batch_size,
            "meta_path": config.meta_path,
            "gpu_graph_training": not self.is_predict,
            "sage_mode": sage_mode,
            "samples": str_samples,
        }

        first_node_type = util.get_first_node_type(config.meta_path)
        graph_config["first_node_type"] = first_node_type

        if self.is_predict:
            graph_config["batch_size"] = config.infer_batch_size
            graph_config["samples"] = str_infer_samples

        dataset = fluid.DatasetFactory().create_dataset("InMemoryDataset")
        dataset.set_feed_type("SlotRecordInMemoryDataFeed")
        dataset.set_use_var(self.holder_list)
        dataset.set_graph_config(graph_config)

        dataset.set_batch_size(1)  # Fixed Don't Change. Batch Size is not config here.
        dataset.set_thread(len(get_cuda_places()))

        dataset.set_hdfs_config(fs_name, fs_ugi)

        dataset._set_use_ps_gpu(self.embedding.parameter_server)
        file_list = self.get_file_list(chunk_index) 
        dataset.set_filelist(file_list)
        with open("datafeed.pbtxt", "w") as fout:
            fout.write(dataset.desc())
        return dataset

    def chunk_generator(self):
        raise NotImplementedError

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
            self.preload_thread_func(dataset_list)
        except Exception as e:
            self.could_load_sem.release()
            self.ins_ready_sem.release()
            log.warning('preload_thread exception :%s' % (e))

    def preload_thread_func(self, dataset_list):
        """ Generate pass dataset asynchronously
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
                dataset = self.generate_dataset(self.config, index)
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

        super(UnsupReprLearningDataset, self).__init__(chunk_num=chunk_num,
                      config=dataset_config,
                      holder_list=holder_list,
                      embedding=embedding,
                      graph=graph,
                      is_predict=False)
    
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


class InferDataset(BaseDataset):
    def __init__(self, chunk_num, dataset_config, holder_list, embedding=None, graph=None):

        self.dataset_config = dataset_config
        super(InferDataset, self).__init__(chunk_num=chunk_num,
                      config=dataset_config,
                      holder_list=holder_list,
                      embedding=embedding,
                      graph=graph,
                      is_predict=True)

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
