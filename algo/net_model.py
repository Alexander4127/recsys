from collections import defaultdict
import numpy as np
import pandas as pd
import spark

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from base import Model
from utils import create_spark
from nets import RecNet, HybridNet


BATCH_SIZE = 16
MAX_LEN = 10


class NetModel(Model):
    def __init__(self, model, optimizer, n_classes, n_users, user_embeds, item_embeds):
        self.user_embeds = user_embeds
        self.item_embeds = item_embeds
        self.history = defaultdict(torch.Tensor)
        self.model = model
        self.optimizer = optimizer
        self.criterion = nn.MSELoss()

    def fit(self, log):
        pass

    def predict(self, users, items, log=None, k=1):
        users = users.toPandas()
        items = items.toPandas()
        n_users = len(users)
        n_items = len(items)

        users_index = np.array(np.repeat(users["user_idx"], n_items))
        item_index = np.array(np.repeat(items["item_idx"], n_users)).reshape([n_items, n_users]).T.reshape(-1)

        probs = self.evaluate(pd.DataFrame({"user_idx": users_index, "item_idx": item_index})).reshape([n_users, n_items])
        probs /= probs.sum(axis=1).reshape(-1, 1)

        recs_ind = np.argsort(probs, axis=1)[:, -k:]
        recs = np.array(items["item_idx"][recs_ind.reshape(-1)]).reshape([n_users, k])
        relevance = probs[np.repeat(np.arange(n_users), k), recs_ind.astype(int).reshape(-1)]

        return create_spark(users, recs.astype(int), relevance, k)

    def get_history(self, user_id):
        bos = torch.zeros(1, self.item_embeds.shape[1])
        if user_id not in self.history:
            return bos
        return torch.cat([bos, torch.tensor(self.item_embeds.loc[self.history[user_id]].to_numpy().astype(float))])

    @torch.no_grad()
    def evaluate(self, log):
        log = log.reset_index(drop=True)
        loader = DataLoader(log.index, batch_size=BATCH_SIZE, shuffle=False)
        all_probs = []
        for index_batch in loader:
            item_idx = np.array(log.loc[index_batch, "item_idx"])
            user_idx = np.array(log.loc[index_batch, "user_idx"])
            probs = self.model(
                torch.tensor(self.user_embeds.loc[user_idx].to_numpy(), dtype=torch.float32),
                torch.tensor(self.item_embeds.loc[item_idx].to_numpy(), dtype=torch.float32),
                pad_sequence([self.get_history(user_id) for user_id in user_idx], batch_first=True)
            )
            all_probs.append(probs)
        return np.concatenate(all_probs)

    def train(self, log):
        log = log.reset_index(drop=True)
        loader = DataLoader(log.index, batch_size=BATCH_SIZE, shuffle=True)
        for index_batch in loader:
            item_idx = np.array(log.loc[index_batch, "item_idx"])
            user_idx = np.array(log.loc[index_batch, "user_idx"])
            response = np.array(log.loc[index_batch, "response"])
            probs = self.model(
                torch.tensor(self.user_embeds.loc[user_idx].to_numpy(), dtype=torch.float32),
                torch.tensor(self.item_embeds.loc[item_idx].to_numpy(), dtype=torch.float32),
                pad_sequence([self.get_history(user_id) for user_id in user_idx], batch_first=True)
            )
            loss = self.criterion(probs.squeeze(), torch.tensor(response).float())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def refit(self, log):
        log = log.toPandas().sort_values(by="user_idx")
        users = np.unique(log["user_idx"])
        k = len(log) // len(users)
        assert k * len(users) == len(log)
        new_recs = np.array(log["item_idx"]).reshape([len(users), k])
        positive_recs = np.array(log["response"]).reshape([len(users), k]).astype(bool)
        for user_idx, rec, pos_mask in zip(users, new_recs, positive_recs):
            self.history[user_idx] = torch.cat([self.history[user_idx], torch.tensor(rec[pos_mask], dtype=torch.float32)])[-MAX_LEN:]
        self.train(log)

    @property
    def item_popularity(self):
        raise NotImplementedError()
