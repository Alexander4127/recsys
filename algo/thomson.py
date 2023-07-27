import numpy as np
import pandas as pd
import spark

from base import Model
from utils import create_spark


class ThomsonModel(Model):
    def __init__(self, n_bandits, foo=lambda x: x):
        self.success_count = np.ones(n_bandits)
        self.failure_count = np.ones(n_bandits)
        self.func = foo

    def fit(self, log):
        pass

    def refit(self, log):
        log = log.toPandas()
        grouped_df = log.groupby("item_idx")
        item_indexes = np.unique(log["item_idx"])
        success_count = np.array(grouped_df.sum().sort_index()["response"])
        self.success_count[item_indexes] += success_count
        self.failure_count[item_indexes] += np.array(grouped_df.count().sort_index()["user_idx"]) - success_count

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
