# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved
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
This file preprocess the FlickrDataset for LINE model.
"""
import argparse
import operator
import os


def process_data(groupsMemberships_file, flickr_links_file, users_label_file,
                 edges_file, users_file):
    """Preprocess flickr network dataset.

    Args:
        groupsMemberships_file: flickr-groupmemberships.txt file, 
            each line is a pair (user, group), which indicates a user belongs to a group.  

        flickr_links_file: flickr-links.txt file,
            each line is a pair (user, user), which indicates 
            the two users have a relationship.

        users_label_file: each line is a pair (user, list of group),
            each user may belong to multiple groups.

        edges_file: each line is a pair (user, user), which indicates 
            the two users have a relationship. It filters some unused edges.

        users_file: each line is a int number, which indicates the ID of a user.
    """
    group2users = {}
    with open(groupsMemberships_file, 'r') as f:
        for line in f:
            user, group = line.strip().split()
            try:
                group2users[int(group)].append(user)
            except:
                group2users[int(group)] = [user]

    # counting how many users belong to every group
    group2usersNum = {}
    for key, item in group2users.items():
        group2usersNum[key] = len(item)

    groups_sorted_by_usersNum = sorted(
        group2usersNum.items(), key=operator.itemgetter(1), reverse=True)

    # the paper only need the 5 groups with the largest number of users
    label = 1  # remapping the 5 groups from 1 to 5
    users_label = {}
    for i in range(5):
        users_list = group2users[groups_sorted_by_usersNum[i][0]]
        for user in users_list:
            # one user may have multi-labels
            try:
                users_label[user].append(label)
            except:
                users_label[user] = [label]
        label += 1

    # remapping the users IDs to make the IDs from 0 to N
    userID2nodeID = {}
    count = 1
    for key in sorted(users_label.keys()):
        userID2nodeID[key] = count
        count += 1

    with open(users_label_file, 'w') as writer:
        for key in sorted(users_label.keys()):
            line = ' '.join([str(i) for i in users_label[key]])
            writer.write(str(userID2nodeID[key]) + ',' + line + '\n')

    # produce edges file
    with open(flickr_links_file, 'r') as reader, open(edges_file,
                                                      'w') as writer:
        for line in reader:
            src, dst = line.strip().split('\t')
            # filter unused user IDs
            if src in users_label and dst in users_label:
                # remapping the users IDs
                src = userID2nodeID[src]
                dst = userID2nodeID[dst]

                writer.write(str(src) + '\t' + str(dst) + '\n')

    # produce nodes file
    with open(users_file, 'w') as writer:
        for i in range(1, 1 + len(userID2nodeID)):
            writer.write(str(i) + '\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='LINE')
    parser.add_argument(
        '--groupmemberships',
        type=str,
        default='./data/flickr/flickr-groupmemberships.txt',
        help='groupmemberships of flickr dataset')

    parser.add_argument(
        '--flickr_links',
        type=str,
        default='./data/flickr/flickr-links.txt',
        help='the flickr-links.txt file for training')

    parser.add_argument(
        '--nodes_label',
        type=str,
        default='./data/flickr/nodes_label.txt',
        help='nodes (users) label file for training')

    parser.add_argument(
        '--edges',
        type=str,
        default='./data/flickr/edges.txt',
        help='the result edges (links) file for training')

    parser.add_argument(
        '--nodes',
        type=str,
        default='./data/flickr/nodes.txt',
        help='the nodes (users) file for training')

    args = parser.parse_args()
    process_data(args.groupmemberships, args.flickr_links, args.nodes_label,
                 args.edges, args.nodes)
