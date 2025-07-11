import os

import numpy as np
import torch

from torch.nn import Module, Embedding, LSTM, Linear, Dropout
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

class DKT(Module):
    def __init__(self, num_c, emb_size, dropout=0.1, emb_type='qid', emb_path="", pretrain_dim=768):
        super().__init__()
        self.model_name = "dkt"
        self.num_c = 6
        self.emb_size = emb_size
        self.hidden_size = emb_size
        self.emb_type = emb_type

        self.interaction_emb = Embedding(self.num_c * 2, self.emb_size)

        self.lstm_layer = LSTM(self.emb_size, self.hidden_size, batch_first=True)
        self.dropout_layer = Dropout(dropout)
        self.out_layer = Linear(self.hidden_size, self.num_c)

    def forward(self, q, r):
        x = q
        self.interaction_emb = Embedding(835, 256).to(device)
        xemb = self.interaction_emb(x)
        self.lstm_layer = LSTM(256, 256).to(device)
        self.dropout_layer = Dropout(0.1).to(device)
        self.out_layer = Linear(256, 835).to(device)
        h, _ = self.lstm_layer(xemb)
        h = self.dropout_layer(h)
        y = self.out_layer(h)
        y = torch.sigmoid(y)

        return y
