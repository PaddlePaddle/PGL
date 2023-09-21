#-*- coding: utf-8 -*-
import argparse
import numpy as np
np.random.seed(123)
from scipy.cluster.vq import vq, kmeans, whiten
from sklearn import metrics

import pgl


def load_graph_embedding(filename):
    embed = []
    with open(filename, "r") as f:
        for line in f:
            nid, vec = line.strip().split("\t")
            vec_list = list(map(float, vec.split(" ")))
            embed.append(vec_list)

    return np.array(embed, dtype="float32")


def kmeans_cluster(args):
    dataset = pgl.dataset.CoraDataset()
    graph_embedding = load_graph_embedding(args.embed_file)
    y = dataset.y
    num_classes = dataset.num_classes

    print("%s number of classes" % num_classes)
    print("%s number nodes" % len(y))
    centroids, _ = kmeans(graph_embedding, num_classes)
    result, _ = vq(graph_embedding, centroids)
    acc1 = metrics.accuracy_score(y, result)

    node_feat = dataset.graph.node_feat["words"]
    node_feat = whiten(node_feat)
    centroids, _ = kmeans(node_feat, num_classes)
    result, _ = vq(node_feat, centroids)
    acc2 = metrics.accuracy_score(y, result)

    print("The accuracy of graph embedding is %.4f" % (acc1))
    print("The accuracy of node features is %.4f" % (acc2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='kmeans')
    parser.add_argument("--embed_file", type=str, default="./embedding.txt")
    args = parser.parse_args()
    kmeans_cluster(args)
