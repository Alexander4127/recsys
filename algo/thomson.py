import collections
import numpy as np
import pandas as pd
import spark

from base import Model
from utils import create_spark


class ThomsonModel(Model):
    def __init__(self, n_bandits, buff_size=None, foo=lambda x: x):
        self.success_count = np.ones(n_bandits)
        self.failure_count = np.ones(n_bandits)
        self.buff_size = buff_size
        self.dq_success_count = collections.deque()
        self.dq_failure_count = collections.deque()
        self.func = foo 
        
    def fit(self, log):
        pass

    def refit(self, log):
        log = log.toPandas()
        grouped_df = log.groupby("item_idx")
        item_indexes = np.unique(log["item_idx"])

        success_count = np.array(grouped_df.sum().sort_index()["response"])

        full_success_count = np.zeros_like(self.success_count)
        full_success_count[item_indexes] = success_count
        self.success_count += full_success_count

        if self.buff_size:
            self.dq_success_count.append(full_success_count)
            if len(self.dq_success_count) > self.buff_size:
                old_full_success = self.dq_success_count.popleft()
                self.success_count -= old_full_success

        failure_count = np.array(grouped_df.count().sort_index()["user_idx"]) - success_count

        full_failure_count = np.zeros_like(self.failure_count)
        full_failure_count[item_indexes] = failure_count
        self.failure_count += full_failure_count

        if self.buff_size:
            self.dq_failure_count.append(full_failure_count)
            if len(self.dq_failure_count) > self.buff_size:
                old_full_failure_count = self.dq_failure_count.popleft()
                self.failure_count -= old_full_failure_count

    def predict(self, users, items, log=None, k=1):
        users = users.toPandas()
        items = items.toPandas()

        samples_list = np.array([
            np.random.beta(
                self.func(1 + self.success_count[bandit_id]),
                self.func(1 + self.failure_count[bandit_id]),
                size=len(users)
            ) for bandit_id in range(len(items))
        ]).T
        samples_list /= samples_list.sum(axis=1).reshape(-1, 1)

        recs = np.argsort(samples_list, axis=1)[:, -k:]
        relevance = samples_list[np.repeat(np.arange(len(users)), k), recs.astype(int).reshape(-1)]

        return create_spark(users, recs.astype(int), relevance, k)

    @property
    def item_popularity(self):
        return list(self.success_count / (self.failure_count + self.success_count))        
