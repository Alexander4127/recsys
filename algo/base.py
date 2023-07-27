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

    @property
    def item_popularity(self):
        raise NotImplementedError()
