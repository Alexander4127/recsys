import numpy as np
import pandas as pd
import spark

from base import Model
from utils import create_spark


class EpsilonGreedyModel(Model):
    def __init__(self, epsilon, n_bandits):
        self.epsilon = epsilon
        self.success_count = np.ones(n_bandits)
        self.count = np.ones(n_bandits)

    def fit(self, log):
        pass

    def refit(self, log):
        log = log.toPandas()
        grouped_df = log.groupby("item_idx")
        item_indexes = np.unique(log["item_idx"])
        self.success_count[item_indexes] += np.array(grouped_df.sum().sort_index()["response"])
        self.count[item_indexes] += np.array(grouped_df.count().sort_index()["user_idx"])

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
