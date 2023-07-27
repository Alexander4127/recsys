import collections
import numpy as np
import pandas as pd
import spark

from base import Model
from utils import create_spark


class EpsilonGreedyModel(Model):
    def __init__(self, epsilon, n_bandits, buff_size=None):
        self.epsilon = epsilon
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
        success_ratio /= np.sum(success_ratio)
        max_rec = np.argsort(success_ratio)[-k:]
        recs = np.ones([len(users), k]) * max_rec

        probs = np.random.random(size=len(users))
        random_mask = probs < self.epsilon
        random_samples = np.array([
            np.random.choice(
                np.arange(len(items)),
                size=k,
                replace=False
            ) for _ in range(len(users))
        ])
        recs[random_mask] = random_samples[random_mask]

        return create_spark(users, recs.astype(int), success_ratio[recs.astype(int)], k)


    @property
    def item_popularity(self):
        return list(self.success_count / self.count)
