import torch
import torch.nn as nn


class RecNet(nn.Module):
    def __init__(self, user_embed_dim, item_embed_dim, hidden_dim=8):
        super().__init__()
        self.lstm = nn.LSTM(input_size=item_embed_dim, hidden_size=item_embed_dim, batch_first=True)
        self.user_model = nn.Sequential(
            nn.Linear(user_embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.result_model = nn.Sequential(
            nn.Linear(hidden_dim + item_embed_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def rec_forward(self, embeds):
        outputs, hidden = self.lstm(embeds.float())
        return outputs[:, -1, :]

    def forward(self, user_embed, item_embed, history_embed):
        rec_embed = self.rec_forward(history_embed)
        user_output = self.user_model(user_embed.float())
        common_input = torch.cat([user_output, item_embed, rec_embed], dim=1)
        return self.result_model(common_input.float())

class HybridNet(nn.Module):
    def __init__(self, user_embed_dim, item_embed_dim, hidden_dim=8):
        super().__init__()
        self.user_model = nn.Sequential(
            nn.Linear(user_embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.result_model = nn.Sequential(
            nn.Linear(hidden_dim + item_embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, user_embed, item_embed, history_embed):
        user_output = self.user_model(user_embed.float())
        common_input = torch.cat([user_output, item_embed], dim=1)
        return self.result_model(common_input.float())
