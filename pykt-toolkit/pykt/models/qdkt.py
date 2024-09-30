import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .que_base_model import QueBaseModel
from pykt.utils import debug_print

# Import our QueEmbedder we implemented for akt_queue
class QueEmbedder(nn.Module):
    # This module decides 1) whether to read or initialize embeddings
    # 2) freeze or train embeddings 
    # 3) apply dim. reduction or not (via one linear layer.)
    def __init__(self, num_q, emb_size, emb_path, flag_load_emb, flag_emb_freezed, model_name):
        super().__init__()
        """
        Input:
            num_q: number of questions
            emb_size: size of embeddings (if different from provided embeddings, they will be cast to emb_size)
            emb_path: path of embeddings to be read from
            flag_load_emb: if TRUE, embeddings will be loaded from the path.
            flag_emb_freezed: if TRUE, embeddings will be fixed, i.e. won't be trained
            model_name: the name of original algorithm that calls this class (mostly for debugging purposes.)
        """
        self.num_q = num_q
        self.emb_size = emb_size
        self.emb_path = emb_path
        self.flag_load_emb = flag_load_emb
        self.flag_emb_freezed = flag_emb_freezed
        self.model_name = model_name

        # After initializing the embedding layer, this value can change, which signals the need of linear projection.
        self.loaded_emb_size = emb_size

        # Initialize embedding layer
        self.init_embedding_layer()

        if self.emb_size != self.loaded_emb_size:
            debug_print(f"Loaded embeddings' size is different than provided emb size. Linear layer will be applied.",fuc_name=self.model_name)
            self.projection_layer = nn.Linear(self.loaded_emb_size, self.emb_size)

        # For debug, count number of trainable params
        num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        debug_print(f"Que Embedder num trainable parameters: {num_params}",fuc_name=self.model_name)

    def init_embedding_layer(self):
        if self.emb_path == '' or not self.flag_load_emb:
            # If flag_emb_freezed is True, these will be fixed, i.e. no grad applied.
            if self.flag_emb_freezed:
                debug_print(f"Embeddings are randomly initialized and freezed",fuc_name=self.model_name)
                self.que_emb = nn.Embedding(self.num_q*2, self.emb_size, _freeze=True)
                # If flag_emb_freezed is False, the grad decent will be applied as usual
            else:
                debug_print(f"Embeddings are randomly initialized and not freezed",fuc_name=self.model_name)
                self.que_emb = nn.Embedding(self.num_q*2, self.emb_size, _freeze=False)

        # If path is not empty, and coming from inder KC pipeline
        # elif 'infer_kc' in self.emb_path and self.flag_load_emb:
        elif self.flag_load_emb:
            with open(self.emb_path, 'r') as f:
                precomputed_embeddings = json.load(f)
            precomputed_embeddings_tensor = torch.tensor([precomputed_embeddings[str(i)] for i in range(len(precomputed_embeddings))], dtype=torch.float)

            # IMPORTANT:
            # emb_size should be changed based on the loaded embeddings!
            num_q_precomputed, self.loaded_emb_size = precomputed_embeddings_tensor.shape # (Num questions x emb size)

            assert self.num_q == num_q_precomputed

            # We get [0, emb] for incorrect, and [emb, 0] for correct answer as initialization
            self.loaded_emb_size = self.loaded_emb_size * 2
            zero_tensor = torch.zeros_like(precomputed_embeddings_tensor)
            precomputed_embeddings_incorrect = torch.cat([zero_tensor, precomputed_embeddings_tensor], dim=1)
            precomputed_embeddings_correct = torch.cat([precomputed_embeddings_tensor, zero_tensor], dim=1)
            precomputed_embeddings_tensor_all = torch.cat([precomputed_embeddings_incorrect, precomputed_embeddings_correct], dim=0)

            # For debug
            orig_norm = precomputed_embeddings_tensor_all[0].norm()
            debug_print(f"The original norm of the embeddings provided is {orig_norm} .",fuc_name=self.model_name)

            # Normalize the lengths to 1, for convenience.
            norms = precomputed_embeddings_tensor_all.norm(p=2, dim=1, keepdim=True)
            precomputed_embeddings_tensor_all = precomputed_embeddings_tensor_all/norms

            # Now scale to expected size.
            precomputed_embeddings_tensor_all = precomputed_embeddings_tensor_all * np.sqrt(self.loaded_emb_size)

            # For debug
            cur_norm = precomputed_embeddings_tensor_all[0].norm()
            debug_print(f"The norm of the embeddings are now scaled to {cur_norm} .",fuc_name=self.model_name)

            # If flag_emb_freezed is True, these will be fixed, i.e. no grad applied.
            if self.flag_emb_freezed:
                debug_print(f"Embeddings are loaded from path and freezed",fuc_name=self.model_name)
                self.que_emb = nn.Embedding.from_pretrained(precomputed_embeddings_tensor_all, freeze=True)
                # If flag_emb_freezed is False, the grad decent will be applied as usual
            else:
                debug_print(f"Embeddings are loaded from path and not freezed",fuc_name=self.model_name)
                self.que_emb = nn.Embedding.from_pretrained(precomputed_embeddings_tensor_all, freeze=False)

        else:
            self.que_emb = nn.Embedding(self.num_q*2, self.emb_size)
            print("Not using the provided path " + emb_path)

    def forward(self, q):
        # It just takes question ids and return (projected) embeddings
        x = self.que_emb(q)
        if self.emb_size != self.loaded_emb_size:
            x = self.projection_layer(x)
        return x



class QDKTNet(nn.Module):
    def __init__(self, num_q,num_c,emb_size, dropout=0.1, emb_type='qaid', emb_path="", flag_load_emb=False, flag_emb_freezed=False, pretrain_dim=768,device='cpu',mlp_layer_num=1,other_config={}):
        super().__init__()
        self.model_name = "qdkt"
        self.num_q = num_q
        self.num_c = num_c
        self.emb_size = emb_size
        self.hidden_size = emb_size
        self.device = device
        self.emb_type = emb_type

        self.que_emb = QueEmbedder(num_q, emb_size, emb_path, flag_load_emb, flag_emb_freezed, self.model_name)
        
        self.lstm_layer = nn.LSTM(self.emb_size, self.hidden_size, batch_first=True)
        self.dropout_layer = nn.Dropout(dropout)
        self.out_layer = nn.Linear(self.hidden_size, self.num_q)


    def forward(self, q, c ,r,data=None):
        x = (q + self.num_q * r)[:,:-1]
        xemb = self.que_emb(x)
        h, _ = self.lstm_layer(xemb)
        h = self.dropout_layer(h)
        y = self.out_layer(h)
        y = torch.sigmoid(y)
        y = (y * F.one_hot(data['qshft'].long(), self.num_q)).sum(-1)
        outputs = {"y":y}
        return outputs

class QDKT(QueBaseModel):
    def __init__(self, num_q,num_c, emb_size, dropout=0.1, emb_type='qaid', emb_path="", flag_load_emb=False, flag_emb_freezed=False, pretrain_dim=768,device='cpu',seed=0,mlp_layer_num=1,other_config={},**kwargs):
        model_name = "qdkt"
       
        debug_print(f"emb_type is {emb_type}",fuc_name="QDKT")

        super().__init__(model_name=model_name,emb_type=emb_type,emb_path=emb_path,pretrain_dim=pretrain_dim,device=device,seed=seed)
        self.model = QDKTNet(num_q=num_q,num_c=num_c,emb_size=emb_size,dropout=dropout,emb_type=emb_type, flag_load_emb=flag_load_emb, flag_emb_freezed=flag_emb_freezed,
                               emb_path=emb_path,pretrain_dim=pretrain_dim,device=device,mlp_layer_num=mlp_layer_num,other_config=other_config)
       
        self.model = self.model.to(device)
        self.emb_type = self.model.emb_type
        self.loss_func = self._get_loss_func("binary_crossentropy")
       
    def train_one_step(self,data,process=True,return_all=False, weighted_loss=0):
        outputs,data_new = self.predict_one_step(data,return_details=True,process=process)
        loss = self.get_loss(outputs['y'],data_new['rshft'],data_new['sm'], weighted_loss=weighted_loss)
        return outputs['y'],loss#y_question没用

    def predict_one_step(self,data,return_details=False,process=True,return_raw=False):
        data_new = self.batch_to_device(data,process=process)
        outputs = self.model(data_new['cq'].long(),data_new['cc'],data_new['cr'].long(),data=data_new)
        if return_details:
            return outputs,data_new
        else:
            return outputs['y']