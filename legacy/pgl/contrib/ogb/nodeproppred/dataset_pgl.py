# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
"""NodePropPredDataset for pgl
"""
import pandas as pd
import shutil, os
import os.path as osp
import numpy as np
from ogb.utils.url import decide_download, download_url, extract_zip
from ogb.nodeproppred import make_master_file  # create master.csv
from pgl.contrib.ogb.io.read_graph_pgl import read_csv_graph_pgl


def to_bool(value):
    """to_bool"""
    return np.array([value], dtype="bool")[0]


class PglNodePropPredDataset(object):
    """PglNodePropPredDataset
    """

    def __init__(self, name, root="dataset"):
        self.name = name  ## original name, e.g., ogbn-proteins
        self.dir_name = "_".join(
            name.split("-")
        ) + "_pgl"  ## replace hyphen with underline, e.g., ogbn_proteins_pgl

        self.original_root = root
        self.root = osp.join(root, self.dir_name)

        self.meta_info = make_master_file.df  #pd.read_csv(
        #os.path.join(os.path.dirname(__file__), "master.csv"), index_col=0)
        if not self.name in self.meta_info:
            error_mssg = "Invalid dataset name {}.\n".format(self.name)
            error_mssg += "Available datasets are as follows:\n"
            error_mssg += "\n".join(self.meta_info.keys())
            raise ValueError(error_mssg)

        self.download_name = self.meta_info[self.name][
            "download_name"]  ## name of downloaded file, e.g., tox21

        self.num_tasks = int(self.meta_info[self.name]["num tasks"])
        self.task_type = self.meta_info[self.name]["task type"]

        super(PglNodePropPredDataset, self).__init__()

        self.pre_process()

    def pre_process(self):
        """pre_process downlaoding data
        """
        processed_dir = osp.join(self.root, 'processed')
        pre_processed_file_path = osp.join(processed_dir, 'pgl_data_processed')

        if osp.exists(pre_processed_file_path):
            # TODO: Reload Preprocess files 
            pass
        else:
            ### check download
            if not osp.exists(osp.join(self.root, "raw", "edge.csv.gz")):
                url = self.meta_info[self.name]["url"]
                if decide_download(url):
                    path = download_url(url, self.original_root)
                    extract_zip(path, self.original_root)
                    os.unlink(path)
                    # delete folder if there exists
                    try:
                        shutil.rmtree(self.root)
                    except:
                        pass
                    shutil.move(
                        osp.join(self.original_root, self.download_name),
                        self.root)
                else:
                    print("Stop download.")
                    exit(-1)

            raw_dir = osp.join(self.root, "raw")

            ### pre-process and save
            add_inverse_edge = to_bool(self.meta_info[self.name][
                "add_inverse_edge"])
            self.graph = read_csv_graph_pgl(
                raw_dir, add_inverse_edge=add_inverse_edge)

            ### adding prediction target
            node_label = pd.read_csv(
                osp.join(raw_dir, 'node-label.csv.gz'),
                compression="gzip",
                header=None).values
            if "classification" in self.task_type:
                node_label = np.array(node_label, dtype=np.int64)
            else:
                node_label = np.array(node_label, dtype=np.float32)

            label_dict = {"labels": node_label}

            # TODO: SAVE preprocess graph
            self.labels = label_dict['labels']

    def get_idx_split(self):
        """Train/Validation/Test split
        """
        split_type = self.meta_info[self.name]["split"]
        path = osp.join(self.root, "split", split_type)

        train_idx = pd.read_csv(
            osp.join(path, "train.csv.gz"), compression="gzip",
            header=None).values.T[0]
        valid_idx = pd.read_csv(
            osp.join(path, "valid.csv.gz"), compression="gzip",
            header=None).values.T[0]
        test_idx = pd.read_csv(
            osp.join(path, "test.csv.gz"), compression="gzip",
            header=None).values.T[0]

        return {
            "train": np.array(
                train_idx, dtype="int64"),
            "valid": np.array(
                valid_idx, dtype="int64"),
            "test": np.array(
                test_idx, dtype="int64")
        }

    def __getitem__(self, idx):
        assert idx == 0, "This dataset has only one graph"
        return self.graph[idx], self.labels

    def __len__(self):
        return 1

    def __repr__(self):  # pragma: no cover
        return '{}({})'.format(self.__class__.__name__, len(self))


if __name__ == "__main__":
    pgl_dataset = PglNodePropPredDataset(name="ogbn-proteins")
    splitted_index = pgl_dataset.get_idx_split()
    print(pgl_dataset[0])
    print(splitted_index)
