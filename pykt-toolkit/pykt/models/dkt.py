import os
import json

import numpy as np
import torch
import torch.nn as nn

from torch.nn import Module, Embedding, LSTM, Linear, Dropout

# class DKT(Module):
#     def __init__(self, num_c, emb_size, dropout=0.1, emb_type='qid', emb_path="", pretrain_dim=768, **kwargs):
#         super().__init__()
#         self.model_name = "dkt"
#         self.num_c = num_c
#         self.emb_size = emb_size
#         self.hidden_size = emb_size
#         self.emb_type = emb_type
#         self.emb_path = emb_path

#         if emb_type.startswith("qid"):
#             # If path is empty, initialize random Embeddings
#             if emb_path == '':
#                 self.interaction_emb = Embedding(self.num_c * 2, self.emb_size)

#             # If path is not empty, and coming from inder KC pipeline
#             if 'infer_kc' in emb_path:
#                 with open(emb_path, 'r') as f:
#                     precomputed_embeddings = json.load(f)
#                 precomputed_embeddings_tensor = torch.tensor([precomputed_embeddings[str(i)] for i in range(len(precomputed_embeddings))], dtype=torch.float)

#                 # IMPORTANT:
#                 # num_c and emb_size should be changed based on the loaded embeddings!
#                 self.num_c, self.emb_size = precomputed_embeddings_tensor.shape # (Num questions x emb size)

#                 # DKT has 2 embeddings for each question: one for correct, one for incorrect answer
#                 # We get -1*emb for incorrect, and original emb for correct answer as initialization
#                 precomputed_embeddings_tensor_all = torch.cat([-precomputed_embeddings_tensor, precomputed_embeddings_tensor], dim=0)

#                 self.interaction_emb = nn.Embedding.from_pretrained(precomputed_embeddings_tensor_all, freeze=False)

#             else:
#                 raise NotImplementedError("We only consider 'infer_kc' embedding paths.")

#         self.lstm_layer = LSTM(self.emb_size, self.hidden_size, batch_first=True)
#         self.dropout_layer = Dropout(dropout)
#         self.out_layer = Linear(self.hidden_size, self.num_c)
        

#     def forward(self, q, r):
#         # print(f"q.shape is {q.shape}")
#         emb_type = self.emb_type
#         if emb_type == "qid":
#             x = q + self.num_c * r
#             xemb = self.interaction_emb(x)
#         # print(f"xemb.shape is {xemb.shape}")
#         h, _ = self.lstm_layer(xemb)
#         h = self.dropout_layer(h)
#         y = self.out_layer(h)
#         y = torch.sigmoid(y)

#         return y

class DKT(Module):
    def __init__(self, num_c, emb_size, dropout=0.1, emb_type='qid', emb_path="", pretrain_dim=768, **kwargs):
        super().__init__()
        self.model_name = "dkt"
        self.num_c = num_c
        self.emb_size = emb_size
        self.hidden_size = emb_size
        self.emb_type = emb_type

        if emb_type.startswith("qid"):
            self.interaction_emb = Embedding(self.num_c * 2, self.emb_size)

        self.lstm_layer = LSTM(self.emb_size, self.hidden_size, batch_first=True)
        self.dropout_layer = Dropout(dropout)
        self.out_layer = Linear(self.hidden_size, self.num_c)
        

    def forward(self, q, r):
        # print(f"q.shape is {q.shape}")
        emb_type = self.emb_type
        if emb_type == "qid":
            x = q + self.num_c * r
            xemb = self.interaction_emb(x)
        # print(f"xemb.shape is {xemb.shape}")
        h, _ = self.lstm_layer(xemb)
        h = self.dropout_layer(h)
        y = self.out_layer(h)
        y = torch.sigmoid(y)

        return y