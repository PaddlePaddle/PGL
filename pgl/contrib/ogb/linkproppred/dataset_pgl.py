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
"""LinkPropPredDataset for pgl
"""
import pandas as pd
import shutil, os
import os.path as osp
import numpy as np
from ogb.utils.url import decide_download, download_url, extract_zip
from ogb.linkproppred import make_master_file
from pgl.contrib.ogb.io.read_graph_pgl import read_csv_graph_pgl


def to_bool(value):
    """to_bool"""
    return np.array([value], dtype="bool")[0]


class PglLinkPropPredDataset(object):
    """PglLinkPropPredDataset
    """

    def __init__(self, name, root="dataset"):
        self.name = name  ## original name, e.g., ogbl-ppa
        self.dir_name = "_".join(name.split(
            "-")) + "_pgl"  ## replace hyphen with underline, e.g., ogbl_ppa_pgl

        self.original_root = root
        self.root = osp.join(root, self.dir_name)

        self.meta_info = make_master_file.df  #pd.read_csv(os.path.join(os.path.dirname(__file__), "master.csv"), index_col=0)
        if not self.name in self.meta_info:
            print(self.name)
            error_mssg = "Invalid dataset name {}.\n".format(self.name)
            error_mssg += "Available datasets are as follows:\n"
            error_mssg += "\n".join(self.meta_info.keys())
            raise ValueError(error_mssg)

        self.download_name = self.meta_info[self.name][
            "download_name"]  ## name of downloaded file, e.g., ppassoc

        self.task_type = self.meta_info[self.name]["task type"]

        super(PglLinkPropPredDataset, self).__init__()

        self.pre_process()

    def pre_process(self):
        """pre_process downlaoding data
        """
        processed_dir = osp.join(self.root, 'processed')
        pre_processed_file_path = osp.join(processed_dir, 'pgl_data_processed')

        if osp.exists(pre_processed_file_path):
            #TODO: Reload Preprocess files
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

            #TODO: SAVE preprocess graph

    def get_edge_split(self):
        """Train/Validation/Test split
        """
        split_type = self.meta_info[self.name]["split"]
        path = osp.join(self.root, "split", split_type)

        train_idx = pd.read_csv(
            osp.join(path, "train.csv.gz"), compression="gzip",
            header=None).values
        valid_idx = pd.read_csv(
            osp.join(path, "valid.csv.gz"), compression="gzip",
            header=None).values
        test_idx = pd.read_csv(
            osp.join(path, "test.csv.gz"), compression="gzip",
            header=None).values

        if self.task_type == "link prediction":
            target_type = np.int64
        else:
            target_type = np.float32

        return {
            "train_edge": np.array(
                train_idx[:, :2], dtype="int64"),
            "train_edge_label": np.array(
                train_idx[:, 2], dtype=target_type),
            "valid_edge": np.array(
                valid_idx[:, :2], dtype="int64"),
            "valid_edge_label": np.array(
                valid_idx[:, 2], dtype=target_type),
            "test_edge": np.array(
                test_idx[:, :2], dtype="int64"),
            "test_edge_label": np.array(
                test_idx[:, 2], dtype=target_type)
        }

    def __getitem__(self, idx):
        assert idx == 0, "This dataset has only one graph"
        return self.graph[0]

    def __len__(self):
        return 1

    def __repr__(self):  # pragma: no cover
        return '{}({})'.format(self.__class__.__name__, len(self))


if __name__ == "__main__":
    pgl_dataset = PglLinkPropPredDataset(name="ogbl-ppa")
    splitted_edge = pgl_dataset.get_edge_split()
    print(pgl_dataset[0])
    print(splitted_edge)
