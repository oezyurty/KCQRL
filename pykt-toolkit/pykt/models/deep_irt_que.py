import os

import numpy as np
import torch
import torch.nn as nn
from torch.nn import Module, Parameter, Embedding, Linear, Dropout
from torch.nn.init import kaiming_normal_

from .akt_que import QueEmbedder
from .que_base_model import QueBaseModel

class DeepIRTQue(QueBaseModel):
    def __init__(self, num_c, dim_s, size_m, dropout=0.2, emb_type='qid', emb_path="", flag_load_emb=False, flag_emb_freezed=False, pretrain_dim=768,  device='cpu',seed=0, **kwargs):
        model_name = "deep_irt_que"
        super().__init__(model_name=model_name,emb_type=emb_type,emb_path=emb_path,pretrain_dim=pretrain_dim,device=device,seed=seed)
        self.model = DeepIRT(num_c=num_c, dim_s=dim_s, size_m=size_m, dropout=dropout, emb_type=emb_type, emb_path=emb_path, flag_load_emb=flag_load_emb, flag_emb_freezed=flag_emb_freezed, pretrain_dim=768, device=device, seed=seed)
        self.emb_type = self.model.emb_type
        self.loss_func = self._get_loss_func("binary_crossentropy")
       
    def train_one_step(self,data,process=True,return_all=False, weighted_loss=0):
        outputs,data_new = self.predict_one_step(data,return_details=True,process=process)
        loss = self.get_loss(outputs['y'],data_new['rshft'],data_new['sm'], weighted_loss=weighted_loss)
        return outputs['y'],loss#y_question没用

    def predict_one_step(self,data,return_details=False,process=True,return_raw=False):
        data_new = self.batch_to_device(data,process=process)
        # input_q = data_new['cq'][:,:-1].long()
        # input_r = data_new['cr'][:,:-1].long()
        # import pdb; pdb.set_trace()
        y = self.model(data_new['cq'].long(), data_new['cr'].long())
        outputs = {"y":y[:,1:]}
        if return_details:
            return outputs,data_new
        else:
            return outputs["y"]

class DeepIRT(Module):
    def __init__(self, num_c, dim_s, size_m, dropout=0.2, emb_type='qid', emb_path="", flag_load_emb=False, flag_emb_freezed=False, pretrain_dim=768, **kwargs):
        super().__init__()
        self.model_name = "deep_irt_que"
        self.num_c = num_c
        self.dim_s = dim_s
        self.size_m = size_m
        self.emb_type = emb_type

        if emb_type.startswith("qid"):
            self.k_emb_layer = QueEmbedder(self.num_c, self.dim_s, emb_path, flag_load_emb, flag_emb_freezed, self.model_name)
            self.Mk = Parameter(torch.Tensor(self.size_m, self.dim_s))
            self.Mv0 = Parameter(torch.Tensor(self.size_m, self.dim_s))

        kaiming_normal_(self.Mk)
        kaiming_normal_(self.Mv0)

        self.v_emb_layer = Embedding(2, self.dim_s)

        self.f_layer = Linear(self.dim_s * 2, self.dim_s)
        self.dropout_layer = Dropout(dropout)
        self.p_layer = Linear(self.dim_s, 1)

        self.diff_layer = nn.Sequential(Linear(self.dim_s,1),nn.Tanh())
        self.ability_layer = nn.Sequential(Linear(self.dim_s,1),nn.Tanh())

        self.e_layer = Linear(self.dim_s, self.dim_s)
        self.a_layer = Linear(self.dim_s, self.dim_s)

    def forward(self, q, r, qtest=False):
        emb_type = self.emb_type
        batch_size = q.shape[0]
        if emb_type == "qid":
            # x = q + self.num_c * r
            k = self.k_emb_layer(q)#question embedding
            v = k + self.v_emb_layer(r)#q,a embedding
        
        Mvt = self.Mv0.unsqueeze(0).repeat(batch_size, 1, 1)

        Mv = [Mvt]

        w = torch.softmax(torch.matmul(k, self.Mk.T), dim=-1)

        # Write Process
        e = torch.sigmoid(self.e_layer(v))
        a = torch.tanh(self.a_layer(v))

        for et, at, wt in zip(
            e.permute(1, 0, 2), a.permute(1, 0, 2), w.permute(1, 0, 2)
        ):
            Mvt = Mvt * (1 - (wt.unsqueeze(-1) * et.unsqueeze(1))) + \
                (wt.unsqueeze(-1) * at.unsqueeze(1))
            Mv.append(Mvt)

        Mv = torch.stack(Mv, dim=1)

        # Read Process
        f = torch.tanh(
            self.f_layer(
                torch.cat(
                    [
                        (w.unsqueeze(-1) * Mv[:, :-1]).sum(-2),
                        k
                    ],
                    dim=-1
                )
            )
        )
        
        stu_ability = self.ability_layer(self.dropout_layer(f))#equ 12
        que_diff = self.diff_layer(self.dropout_layer(k))#equ 13

        p = torch.sigmoid(3.0*stu_ability-que_diff)#equ 14
        p = p.squeeze(-1)
        if not qtest:
            return p
        else:
            print(f"f shape is {f.shape},k shape is {k.shape}")
            return p, f, k