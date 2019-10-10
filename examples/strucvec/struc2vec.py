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
"""
struc2vec.py
"""
import argparse
import math
import random
import numpy as np
import pgl
from pgl import graph
from pgl.graph_kernel import alias_sample_build_table
from pgl.sample import alias_sample
from data_loader import EdgeDataset
from classify import train_lr_model
from sklearn_classify import train_lr_l2_model


def selectDegrees(degree_root, index_left, index_right, degree_left,
                  degree_right):
    """
    Select the which degree to be next step.
    """

    if index_left == -1:
        degree_now = degree_right
    elif index_right == -1:
        degree_now = degree_left
    elif (abs(degree_left - degree_root) < abs(degree_right - degree_root)):
        degree_now = degree_left
    else:
        degree_now = degree_right

    return degree_now


class StrucVecGraph():
    """
    The class wrapper the PGL graph, the class involve the funtions to implement struc2vec algorithm.
    """

    def __init__(self, graph, nodes, opt1, opt2, opt3, depth, num_walks,
                 walk_depth):
        self.graph = graph
        self.nodes = nodes
        self.opt1 = opt1
        self.opt2 = opt2
        self.opt3 = opt3
        self.num_walks = num_walks
        self.walk_depth = walk_depth
        self.tag = args.tag
        self.degree_list = dict()
        self.degree2nodes = dict()
        self.node2degree = dict()
        self.distance = dict()
        self.degrees_sorted = None
        self.layer_distance = dict()
        self.layer_message = dict()
        self.layer_norm_distance = dict()
        self.sample_alias = dict()
        self.sample_events = dict()
        self.layer_node_weight_count = dict()
        if opt3 == True:
            self.depth = depth
        else:
            self.depth = 1000

    def distance_func(self, a, b):
        """
        The basic function to calculate the distance between two list with different length.
        """
        ep = 0.5
        m = max(a, b) + ep
        mi = min(a, b) + ep
        return ((m / mi) - 1)

    def distance_opt1_func(self, a, b):
        """
        The optimization function to calculate the distance between two list with list count.
        """
        ep = 0.5
        m = max(a[0], b[0]) + ep
        mi = min(a[0], b[0]) + ep
        return ((m / mi) - 1) * max(a[1], b[1])

    def add_degree_todict(self, node_id, degree, depth, opt1):
        """
        output the degree of each node to a dict
        """
        if node_id not in self.degree_list:
            self.degree_list[node_id] = dict()
        if depth not in self.degree_list[node_id]:
            self.degree_list[node_id][depth] = None
        if opt1:
            degree = np.array(np.unique(degree, return_counts=True)).T
        self.degree_list[node_id][depth] = degree

    def output_degree_with_depth(self, depth, opt1):
        """
        according to the BFS to get the degree of each layer
        """
        degree_dict = dict()

        for node in self.nodes:
            start_node = node
            cur_node = node
            cur_dep = 0
            flag_visit = set()
            while cur_node is not None and cur_dep < depth:
                if not isinstance(cur_node, list):
                    cur_node = [cur_node]
                filter_node = []
                for node in cur_node:
                    if node not in flag_visit:
                        flag_visit.add(node)
                        filter_node.append(node)
                cur_node = filter_node
                if len(cur_node) == 0:
                    break
                outdegree = self.graph.outdegree(cur_node)
                mask = (outdegree != 0)
                if np.any(mask):
                    outdegree = np.sort(outdegree[mask])
                else:
                    break
                # save the layer degree message to dict 
                self.add_degree_todict(start_node, outdegree[mask], cur_dep,
                                       opt1)
                succes = self.graph.successor(cur_node)
                cur_node = []
                for succ in succes:
                    if isinstance(succ, np.ndarray):
                        cur_node.extend(succ.flatten().tolist())
                    elif isinstance(succ, int):
                        cur_node.append(succ)
                cur_node = list(set(cur_node))
                cur_dep += 1

    def get_sim_neighbours(self, node, selected_num):
        """
        Select the neighours by using the degree similiarity.
        """
        degree = self.node2degree[node]
        select_count = 0
        node_nbh_list = list()
        for node_nbh in self.degree2nodes[degree]:
            if node != node_nbh:
                node_nbh_list.append(node_nbh)
                select_count += 1
                if select_count > selected_num:
                    return node_nbh_list
        degree_vec_len = len(self.degrees_sorted)
        index_degree = self.degrees_sorted.index(degree)

        index_left = -1
        index_right = -1
        degree_left = -1
        degree_right = -1

        if index_degree != -1 and index_degree >= 1:
            index_left = index_degree - 1
        if index_degree != -1 and index_degree <= degree_vec_len - 2:
            index_right = index_degree + 1
        if index_left == -1 and index_right == -1:
            return node_nbh_list
        if index_left != -1:
            degree_left = self.degrees_sorted[index_left]
        if index_right != -1:
            degree_right = self.degrees_sorted[index_right]
        select_degree = selectDegrees(degree, index_left, index_right,
                                      degree_left, degree_right)
        while True:
            for node_nbh in self.degree2nodes[select_degree]:
                if node_nbh != node:
                    node_nbh_list.append(node_nbh)
                    select_count += 1
                    if select_count > selected_num:
                        return node_nbh_list

            if select_degree == degree_left:
                if index_left >= 1:
                    index_left = index_left - 1
                else:
                    index_left = -1

            else:
                if index_right <= degree_vec_len - 2:
                    index_right += 1
                else:
                    index_right = -1

            if index_left == -1 and index_right == -1:
                return node_nbh_list

            if index_left != -1:
                degree_left = self.degrees_sorted[index_left]
            if index_right != -1:
                degree_right = self.degrees_sorted[index_right]
            select_degree = selectDegrees(degree, index_left, index_right,
                                          degree_left, degree_right)
        return node_nbh_list

    def calc_node_with_neighbor_dtw_opt2(self, src):
        """
        Use the optimization algorithm to reduce the next steps range. 
        """
        from fastdtw import fastdtw
        node_nbh_list = self.get_sim_neighbours(src, self.selected_nbh_nums)
        distance = {}
        for dist in node_nbh_list:
            calc_layer_len = min(len(self.degree_list[src]), \
                len(self.degree_list[dist]))
            distance_iteration = 0.0
            distance[src, dist] = {}
            for layer in range(0, calc_layer_len):
                src_layer = self.degree_list[src][layer]
                dist_layer = self.degree_list[dist][layer]
                weight, path = fastdtw(
                    src_layer,
                    dist_layer,
                    radius=1,
                    dist=self.distance_calc_func)
                distance_iteration += weight
                distance[src, dist][layer] = distance_iteration
        return distance

    def calc_node_with_neighbor_dtw(self, src_index):
        """
        No optimization algorithm to reduce the next steps range, just calculate distance of all path. 
        """
        from fastdtw import fastdtw
        distance = {}
        for dist_index in range(src_index + 1, self.graph.num_nodes - 1):
            src = self.nodes[src_index]
            dist = self.nodes[dist_index]
            calc_layer_len = min(len(self.degree_list[src]), \
                len(self.degree_list[dist]))
            distance_iteration = 0.0
            distance[src, dist] = {}
            for layer in range(0, calc_layer_len):
                src_layer = self.degree_list[src][layer]
                dist_layer = self.degree_list[dist][layer]
                weight, path = fastdtw(
                    src_layer,
                    dist_layer,
                    radius=1,
                    dist=self.distance_calc_func)
                distance_iteration += weight
                distance[src, dist][layer] = distance_iteration
        return distance

    def calc_distances_between_nodes(self):
        """
        Use the dtw algorithm to calculate the distance between nodes. 
        """
        from fastdtw import fastdtw
        from pathos.multiprocessing import Pool
        # decide use which algo to use 
        if self.opt1 == True:
            self.distance_calc_func = self.distance_opt1_func
        else:
            self.distance_calc_func = self.distance_func

        dtws = []
        if self.opt2:
            depth = 0
            for node in self.nodes:
                if node in self.degree_list:
                    if depth in self.degree_list[node]:
                        degree = self.degree_list[node][depth]
                        if args.opt1:
                            degree = degree[0][0]
                        else:
                            degree = degree[0]
                    if degree not in self.degree2nodes:
                        self.degree2nodes[degree] = []
                    if node not in self.node2degree:
                        self.node2degree[node] = degree
                    self.degree2nodes[degree].append(node)
            # select the log(n) node to select data 
            degree_keys = self.degree2nodes.keys()
            degree_keys = np.array(list(degree_keys), dtype='int')
            self.degrees_sorted = list(np.sort(degree_keys))
            selected_nbh_nums = 2 * math.log(self.graph.num_nodes - 1, 2)
            self.selected_nbh_nums = selected_nbh_nums

            pool = Pool(10)
            dtws = pool.map(self.calc_node_with_neighbor_dtw_opt2, self.nodes)
            pool.close()
            pool.join()
        else:
            src_indices = range(0, self.graph.num_nodes - 2)

            pool = Pool(10)
            dtws = pool.map(self.calc_node_with_neighbor_dtw, src_indices)
            pool.close()
            pool.join()
        print('calc the dtw done.')
        for dtw in dtws:
            self.distance.update(dtw)

    def normlization_layer_weight(self):
        """
        Normlation the distance between nodes, weight[1, 2, ....N] = distance[1, 2, ......N] / sum(distance)
        """
        for sd_keys, layer_weight in self.distance.items():
            src, dist = sd_keys
            layers, weights = layer_weight.keys(), layer_weight.values()
            for layer, weight in zip(layers, weights):
                if layer not in self.layer_distance:
                    self.layer_distance[layer] = {}
                if layer not in self.layer_message:
                    self.layer_message[layer] = {}
                self.layer_distance[layer][src, dist] = weight

                if src not in self.layer_message[layer]:
                    self.layer_message[layer][src] = []
                if dist not in self.layer_message[layer]:
                    self.layer_message[layer][dist] = []
                self.layer_message[layer][src].append(dist)
                self.layer_message[layer][dist].append(src)

        # normalization the layer weight  
        for i in range(0, self.depth):
            layer_weight = 0.0
            layer_count = 0
            if i not in self.layer_norm_distance:
                self.layer_norm_distance[i] = {}
            if i not in self.sample_alias:
                self.sample_alias[i] = {}
            if i not in self.sample_events:
                self.sample_events[i] = {}
            if i not in self.layer_message:
                continue
            for node in self.nodes:
                if node not in self.layer_message[i]:
                    continue
                nbhs = self.layer_message[i][node]
                weights = []
                sum_weight = 0.0
                for dist in nbhs:
                    if (node, dist) in self.layer_distance[i]:
                        weight = self.layer_distance[i][node, dist]
                    else:
                        weight = self.layer_distance[i][dist, node]
                    weight = np.exp(-float(weight))
                    weights.append(weight)
                # norm the weight 
                sum_weight = sum(weights)
                if sum_weight == 0.0:
                    sum_weight = 1.0
                weight_list = [weight / sum_weight for weight in weights]
                self.layer_norm_distance[i][node] = weight_list
                alias, events = alias_sample_build_table(np.array(weight_list))
                self.sample_alias[i][node] = alias
                self.sample_events[i][node] = events
                layer_weight += 1.0
                #layer_weight += sum(weight_list)
                layer_count += len(weights)
            layer_avg_weight = layer_weight / (1.0 * layer_count)

            self.layer_node_weight_count[i] = dict()
            for node in self.nodes:
                if node not in self.layer_norm_distance[i]:
                    continue
                weight_list = self.layer_norm_distance[i][node]
                node_cnt = 0
                for weight in weight_list:
                    if weight > layer_avg_weight:
                        node_cnt += 1
                self.layer_node_weight_count[i][node] = node_cnt

    def choose_neighbor_alias_method(self, node, layer):
        """
        Choose the neighhor with strategy of random 
        """
        weight_list = self.layer_norm_distance[layer][node]
        neighbors = self.layer_message[layer][node]
        select_idx = alias_sample(1, self.sample_alias[layer][node],
                                  self.sample_events[layer][node])
        return neighbors[select_idx[0]]

    def choose_layer_to_walk(self, node, layer):
        """
        Choose the layer to random walk
        """
        random_value = random.random()
        higher_neigbours_nums = self.layer_node_weight_count[layer][node]
        prob = math.log(higher_neigbours_nums + math.e)
        prob = prob / (1.0 + prob)
        if random_value > prob:
            if layer > 0:
                layer = layer - 1
        else:
            if layer + 1 in self.layer_message and \
                node in self.layer_message[layer + 1]:
                layer = layer + 1
        return layer

    def executor_random_walk(self, walk_process_id):
        """
        The main function to execute the structual random walk 
        """
        nodes = self.nodes
        random.shuffle(nodes)
        walk_path_all_nodes = []
        for node in nodes:
            walk_path = []
            walk_path.append(node)
            layer = 0
            while len(walk_path) < self.walk_depth:
                prop = random.random()
                if prop < 0.3:
                    node = self.choose_neighbor_alias_method(node, layer)
                    walk_path.append(node)
                else:
                    layer = self.choose_layer_to_walk(node, layer)
            walk_path_all_nodes.append(walk_path)
        return walk_path_all_nodes

    def random_walk_structual_sim(self):
        """
        According to struct distance to walk the path 
        """
        from pathos.multiprocessing import Pool
        print('start process struc2vec random walk.')
        walks_process_ids = [i for i in range(0, self.num_walks)]
        pool = Pool(10)
        walks = pool.map(self.executor_random_walk, walks_process_ids)
        pool.close()
        pool.join()

        #save the final walk result 
        file_result = open(args.tag + "_walk_path", "w")
        for walk in walks:
            for walk_node in walk:
                walk_node_str = " ".join([str(node) for node in walk_node])
                file_result.write(walk_node_str + "\n")
        file_result.close()
        print('process struc2vec random walk done.')


def learning_embedding_from_struc2vec(args):
    """
    Learning the word2vec from the random path
    """
    from gensim.models import Word2Vec
    from gensim.models.word2vec import LineSentence
    struc_walks = LineSentence(args.tag + "_walk_path")
    model = Word2Vec(struc_walks, size=args.w2v_emb_size, window=args.w2v_window_size, iter=args.w2v_epoch, \
        min_count=0, hs=1, sg=1, workers=5)
    model.wv.save_word2vec_format(args.emb_file)


def main(args):
    """
    The main fucntion to run the algorithm struc2vec
    """
    if args.train:
        dataset = EdgeDataset(
            undirected=args.undirected, data_dir=args.edge_file)
        graph = StrucVecGraph(dataset.graph, dataset.nodes, args.opt1, args.opt2, args.opt3, args.depth,\
            args.num_walks, args.walk_depth)
        graph.output_degree_with_depth(args.depth, args.opt1)
        graph.calc_distances_between_nodes()
        graph.normlization_layer_weight()
        graph.random_walk_structual_sim()
        learning_embedding_from_struc2vec(args)
        file_label = open(args.label_file)
        file_label_reindex = open(args.label_file + "_reindex", "w")
        for line in file_label:
            items = line.strip("\n\r").split(" ")
            try:
                items = [int(item) for item in items]
            except:
                continue
            if items[0] not in dataset.node_dict:
                continue
            reindex = dataset.node_dict[items[0]]
            file_label_reindex.write(str(reindex) + " " + str(items[1]) + "\n")
        file_label_reindex.close()

    if args.valid:
        emb_file = open(args.emb_file)
        file_label_reindex = open(args.label_file + "_reindex")
        label_dict = dict()
        for line in file_label_reindex:
            items = line.strip("\n\r").split(" ")
            try:
                label_dict[int(items[0])] = int(items[1])
            except:
                continue

        data_for_train_valid = []
        for line in emb_file:
            items = line.strip("\n\r").split(" ")
            if len(items) <= 2:
                continue
            index = int(items[0])
            label = int(label_dict[index])
            sample = []
            sample.append(index)
            feature_emb = items[1:]
            feature_emb = [float(feature) for feature in feature_emb]
            sample.extend(feature_emb)
            sample.append(label)
            data_for_train_valid.append(sample)
        train_lr_l2_model(args, data_for_train_valid)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='struc2vec')
    parser.add_argument("--edge_file", type=str, default="")
    parser.add_argument("--label_file", type=str, default="")
    parser.add_argument("--emb_file", type=str, default="w2v_emb")
    parser.add_argument("--undirected", type=bool, default=True)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--num_walks", type=int, default=10)
    parser.add_argument("--walk_depth", type=int, default=80)
    parser.add_argument("--opt1", type=bool, default=False)
    parser.add_argument("--opt2", type=bool, default=False)
    parser.add_argument("--opt3", type=bool, default=False)
    parser.add_argument("--w2v_emb_size", type=int, default=128)
    parser.add_argument("--w2v_window_size", type=int, default=10)
    parser.add_argument("--w2v_epoch", type=int, default=5)
    parser.add_argument("--train", type=bool, default=False)
    parser.add_argument("--valid", type=bool, default=False)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--num_class", type=int, default=4)
    parser.add_argument("--epoch", type=int, default=2000)
    parser.add_argument("--tag", type=str, default="")

    args = parser.parse_args()
    main(args)
