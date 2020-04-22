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
Evaluate.py: Evaluator for the results of knowledge graph embeddings.
"""
import numpy as np
import timeit
from mp_mapper import mp_reader_mapper
from pgl.utils.logger import log


class Evaluate:
    """
    Evaluate for trained models.
    """

    def __init__(self, reader):
        self.reader = reader
        self.training_triple_pool = self.reader.training_triple_pool

    @staticmethod
    def rank_extract(results, training_triple_pool):
        """
        :param results: the scores of test examples.
        :param training_triple_pool: existing edges.
        :return: the ranks.
        """
        eval_triple, head_score, tail_score = results
        head_order = np.argsort(head_score)
        tail_order = np.argsort(tail_score)
        head, relation, tail = eval_triple[0], eval_triple[1], eval_triple[2]
        head_rank_raw = 1
        tail_rank_raw = 1
        head_rank_filter = 1
        tail_rank_filter = 1
        for candidate in head_order:
            if candidate == head:
                break
            else:
                head_rank_raw += 1
                if (candidate, relation, tail) in training_triple_pool:
                    continue
                else:
                    head_rank_filter += 1
        for candidate in tail_order:
            if candidate == tail:
                break
            else:
                tail_rank_raw += 1
                if (head, relation, candidate) in training_triple_pool:
                    continue
                else:
                    tail_rank_filter += 1
        return head_rank_raw, tail_rank_raw, head_rank_filter, tail_rank_filter

    def launch_evaluation(self,
                          exe,
                          program,
                          reader,
                          fetch_list,
                          num_workers=4):
        """
        launch_evaluation
        :param exe: executor.
        :param program: paddle program.
        :param reader: test reader.
        :param fetch_list: fetch list.
        :param num_workers: num of workers.
        :return: None
        """

        def func(training_triple_pool):
            """func"""

            def run_func(results):
                """run_func"""
                return self.rank_extract(results, training_triple_pool)

            return run_func

        def iterator():
            """iterator"""
            n_used_eval_triple = 0
            start = timeit.default_timer()
            for batch_feed_dict in reader():
                head_score, tail_score = exe.run(program=program,
                                                 fetch_list=fetch_list,
                                                 feed=batch_feed_dict)
                yield batch_feed_dict["test_triple"], head_score, tail_score
                n_used_eval_triple += 1
                if n_used_eval_triple % 500 == 0:
                    print('[{:.3f}s] #evaluation triple: {}/{}'.format(
                        timeit.default_timer(
                        ) - start, n_used_eval_triple, self.reader.test_num))

        res_reader = mp_reader_mapper(
            reader=iterator,
            func=func(self.training_triple_pool),
            num_works=num_workers)
        self.result(res_reader)

    @staticmethod
    def result(rank_result_iter):
        """
        Calculate the final results.
        :param rank_result_iter: results iter.
        :return: None
        """
        all_rank = [[], []]
        for data in rank_result_iter():
            for i in range(4):
                all_rank[i // 2].append(data[i])

        raw_rank = np.array(all_rank[0])
        filter_rank = np.array(all_rank[1])
        log.info("-----Raw-Average-Results")
        log.info(
            'MeanRank: {:.2f}, MRR: {:.4f}, Hits@1: {:.4f}, Hits@3: {:.4f}, Hits@10: {:.4f}'.
            format(raw_rank.mean(), (1 / raw_rank).mean(), (raw_rank <= 1).
                   mean(), (raw_rank <= 3).mean(), (raw_rank <= 10).mean()))
        log.info("-----Filter-Average-Results")
        log.info(
            'MeanRank: {:.2f}, MRR: {:.4f}, Hits@1: {:.4f}, Hits@3: {:.4f}, Hits@10: {:.4f}'.
            format(filter_rank.mean(), (1 / filter_rank).mean(), (
                filter_rank <= 1).mean(), (filter_rank <= 3).mean(), (
                    filter_rank <= 10).mean()))
