import collections
import numpy as np
import pandas as pd
import spark

from base import Model
from utils import create_spark

class LinUCB(Model):
    def __init__(self, items_features, n_bandits, n_features, C):
        self.feat = items_features.drop(columns=["item_idx"])
        self.feat.index = items_features["item_idx"]
        self.item_idx = np.array(items_features["item_idx"])
        self.inv_As = np.array([np.eye(n_features) for _ in range(n_bandits)])
        self.bs = np.zeros((n_bandits, n_features))
        self.C = C

    def one_rank_update(self, inv_A, x, c):
        # uses Woodbury identity
#         const = 1 / (1 / c + x.T @ inv_A @ x)
#         return inv_A - const * inv_A @ x @ (inv_A.T @ x).T
        return np.linalg.inv(np.linalg.inv(inv_A) + c * x.reshape(-1, 1) * x.T)

    def fit(self, log):
        pass

    def refit(self, log):
        log = log.toPandas()
        for item_idx, real_index in enumerate(self.item_idx):
            assert (log.index == np.arange(len(log))).all()
            n_items = np.sum(log["item_idx"] == real_index)
            n_success = np.sum((log["item_idx"] == real_index) & (log["response"] == 1))
            # x = ... features for `item_idx` item
            x = np.array(self.feat.loc[real_index])
            self.inv_As[item_idx] = self.one_rank_update(self.inv_As[item_idx], x, n_items)
            self.bs[item_idx] += n_success * x

    def predict(self, users, items, log=None, k=1):
        users = users.toPandas()
        items = items.toPandas()

        X = np.array(self.feat)
        thetas = [inv_A @ b for inv_A, b in zip(self.inv_As, self.bs)]
        assert len(X) == len(thetas) == len(items)
        ps = np.array([x.T @ theta + self.C * x.T @ inv_A @ x 
              for x, theta, inv_A in zip(X, thetas, self.inv_As)])
        ps = softmax(ps)
        # thetas = self.inv_As @ self.bs
        # ps = thetas.T @ X + self.C * np.sqrt(X.T @ self.inv_As @ X)

        recs = np.array([
            np.random.choice(
                np.arange(len(items)),
                size=k,
                replace=False,
                p=ps
            ) for _ in range(len(users))
        ], dtype=int)

        relevance = np.repeat(ps, len(users)).reshape([len(items), len(users)]).T[
            np.repeat(np.arange(len(users)), k),
            recs.reshape(-1)
        ]

        recs_idx = self.item_idx[recs.reshape(-1)].reshape([len(users), k])
        return create_spark(users, recs_idx, relevance, k)

    @property
    def item_popularity(self):
        raise NotImplementedError()
