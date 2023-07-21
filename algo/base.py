from abc import ABC, abstractmethod

import spark
import numpy as np


class Model(ABC):
    @abstractmethod
    def fit(self, log):
        raise NotImplementedError()

    @abstractmethod
    def refit(self, log):
        raise NotImplementedError()

    @abstractmethod
    def predict(self, users, items, log=None, k=1):
        raise NotImplementedError()


def createSpark(df, recs, relevance, k):
    recs = recs.reshape(-1)
    repeated_users = np.repeat(df["user_idx"], k)
    df = df.merge(repeated_users, on="user_idx")
    assert len(df) == len(recs), f'{len(df)} != {len(recs)}'
    df["item_idx"] = recs
    df["relevance"] = relevance.reshape(-1)
    return spark.createDataFrame(df)
