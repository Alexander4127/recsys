import collections
import numpy as np
import pandas as pd
import spark

from base import Model
from utils import create_spark


class UCBModel(Model):
    def __init__(self, C, n_bandits, buff_size=None):
        self.C = C
        self.success_count = np.ones(n_bandits)
        self.count = np.ones(n_bandits)
        self.buff_size = buff_size
        self.dq_success_count = collections.deque()
        self.dq_count = collections.deque()

    def fit(self, log):
        pass

    def refit(self, log):
        log = log.toPandas()
        grouped_df = log.groupby("item_idx")
        item_indexes = np.unique(log["item_idx"])
        
        full_success_count = np.zeros_like(self.success_count)
        full_success_count[item_indexes] = np.array(grouped_df.sum().sort_index()["response"])
        self.success_count += full_success_count
        
        if self.buff_size:
            self.dq_success_count.append(full_success_count)
            if len(self.dq_success_count) > self.buff_size:
                old_full_success = self.dq_success_count.popleft()
                self.success_count -= old_full_success
            
        full_count = np.zeros_like(self.count)
        full_count[item_indexes] = np.array(grouped_df.count().sort_index()["user_idx"])
        self.count += full_count
        
        if self.buff_size:
            self.dq_count.append(full_count)
            if len(self.dq_count) > self.buff_size:
                old_full_count = self.dq_count.popleft()
                self.count -= old_full_count

    def predict(self, users, items, log=None, k=1):
        users = users.toPandas()
        items = items.toPandas()

        success_ratio = self.success_count / self.count
        sqrt_term = np.sqrt(2 * np.log(np.sum(self.count)) / self.count)
        relevants = success_ratio + self.C * sqrt_term
        probs = relevants / np.sum(relevants)
        recs = np.array([
            np.random.choice(
                np.arange(len(items)),
                size=k,
                replace=False,
                p=probs
            ) for _ in range(len(users))
        ], dtype=int)

        return create_spark(users, recs, probs[recs], k)

    @property
    def item_popularity(self):
        return list(self.success_count / self.count)
