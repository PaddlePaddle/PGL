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

import pgl
import numpy as np
from collections import defaultdict, Counter
import multiprocessing
import tqdm
import time
import pickle as pkl
import numpy as np
import numpy as np
from collections import defaultdict, Counter
import multiprocessing
import time
import pickle as pkl
import numpy as np

root = "dataset_path"
graph_paper2paper = pgl.Graph.load(
    os.path.join(root, "paper_to_paper_symmetric_pgl_split"))
graph_paper2author = pgl.Graph.load(
    os.path.join(root, "paper_to_author_symmetric_pgl_split_src"))
graph_author2paper = pgl.Graph.load(
    os.path.join(root, "paper_to_author_symmetric_pgl_split_dst"))

labels = np.load(
    os.path.join(root, "mag240_kddcup2021", "processed", "paper",
                 "node_label.npy"),
    mmap_mode="r")


def find_similar_paper_single(pid):
    authors = graph_paper2author.predecessor([pid])[0]
    papers, counts = np.unique(
        np.concatenate(graph_author2paper.predecessor(authors)),
        return_counts=True)
    indegree = graph_paper2author.indegree(papers.tolist())
    indegree = indegree + graph_paper2author.indegree([pid])[0] - counts
    counts = counts / indegree
    # select neighbor with following condition
    #   1. coauthor jaccard  > 0.5 
    #   2. labels are not NaN
    mask = (counts >= 0.5) & (papers != pid) & (
        ~np.isnan(labels[papers.tolist()]))
    papers = papers[mask]
    counts = counts[mask]
    edges = np.vstack([papers, np.ones_like(papers) * pid])
    return edges.T.astype("int64")


def find_similar_paper(start_pid):
    all_edges = []
    for pid in range(start_pid,
                     min(graph_paper2paper.num_nodes, start_pid + batch_size)):
        edges = find_similar_paper_single(pid)
        if len(edges) > 0:
            all_edges.append(edges)
    return np.concatenate(all_edges, dtype="int64")


def get_edge_feat(batch_start):
    batch_size = 5000
    batch_end = min(batch_start + batch_size, graph_paper2paper.num_nodes)
    last_x = None
    last_x_authors = set()
    last_x_papers = set()
    edges = []
    for x in np.arange(batch_start, batch_end):
        edges.append(find_similar_paper(x))
    return np.concatenate(edges)


if __name__ == "__main__":
    batch_size = 5000
    dataset = np.arange(0, graph_paper2paper.num_nodes, step=batch_size)
    max_workers = 20
    edges = []
    with multiprocessing.Pool(max_workers) as pool:
        chunksize = 1000
        imap_unordered_it = pool.imap_unordered(find_similar_paper, dataset,
                                                chunksize)
        start = 0
        for edge_feat in tqdm.tqdm(imap_unordered_it, total=len(dataset)):
            if len(edge_feat) > 0:
                edges.append(edge_feat.astype("int64"))
        edges = np.concatenate(edges, dtype="int64")
        edge_types = np.full([edges.shape[0]], 6, dtype="int32")
        g = pgl.Graph(
            num_nodes=graph_paper2paper.num_nodes,
            edges=edges,
            edge_feat={"edge_type": edge_types})
        g.adj_dst_index
        g.dump(os.path.join(root, "paper_coauthor_paper_symmetric_jc0.5"))
