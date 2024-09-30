import torch
from torch import nn
import numpy as np

from torch.nn import Module, Embedding, Linear, MultiheadAttention, LayerNorm, Dropout
from .utils import transformer_FFN, pos_encode, ut_mask, get_clones

from .akt_que import QueEmbedder
from .qdkt import QueEmbedder as QueAnsEmbedder
from .que_base_model import QueBaseModel


class SAKTQue(QueBaseModel):
    def __init__(self, num_c, seq_len, emb_size, num_attn_heads, dropout, num_en=2, emb_type="qid", emb_path="", flag_load_emb=False, flag_emb_freezed=False, pretrain_dim=768, device='cpu', seed=0, **kwargs):
        model_name = "sakt_que"
        super().__init__(model_name=model_name,emb_type=emb_type,emb_path=emb_path,pretrain_dim=pretrain_dim,device=device,seed=seed)
        self.model = SAKT(num_c=num_c, seq_len=seq_len, 
            emb_size=emb_size, num_attn_heads=num_attn_heads, dropout=dropout, num_en=num_en, 
            emb_type=emb_type, emb_path=emb_path, flag_load_emb=flag_load_emb, flag_emb_freezed=flag_emb_freezed, pretrain_dim=pretrain_dim)
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
        y = self.model(data_new['q'].long(), data_new['r'].long(), data_new['qshft'].long())
        outputs = {"y":y}
        if return_details:
            return outputs,data_new
        else:
            return outputs["y"]

class SAKT(Module):
    def __init__(self, num_c, seq_len, emb_size, num_attn_heads, dropout, num_en=2, emb_type="qid", emb_path="", flag_load_emb=False, flag_emb_freezed=False, pretrain_dim=768, **kwargs):
        super().__init__()
        self.model_name = "sakt_que"
        self.emb_type = emb_type

        self.num_c = num_c
        self.seq_len = seq_len
        self.emb_size = emb_size
        self.num_attn_heads = num_attn_heads
        self.dropout = dropout
        self.num_en = num_en

        if emb_type.startswith("qid"):
            # num_c, seq_len, emb_size, num_attn_heads, dropout, emb_path="")
            # self.interaction_emb = Embedding(num_c * 2, emb_size)
            self.exercise_emb = QueEmbedder(num_c, emb_size, emb_path, flag_load_emb, flag_emb_freezed, self.model_name)
            # self.P = Parameter(torch.Tensor(self.seq_len, self.emb_size))
            # self.qa_embed = nn.Embedding(2, emb_size)
            self.interaction_emb = QueAnsEmbedder(num_c, emb_size, emb_path, flag_load_emb, flag_emb_freezed, self.model_name)
        self.position_emb = Embedding(seq_len, emb_size)

        self.blocks = get_clones(Blocks(emb_size, num_attn_heads, dropout), self.num_en)

        self.dropout_layer = Dropout(dropout)
        self.pred = Linear(self.emb_size, 1)

    def base_emb(self, q, r, qry):
        # qshftemb, xemb = self.exercise_emb(qry), self.exercise_emb(q) + self.qa_embed(r)
    
        # posemb = self.position_emb(pos_encode(xemb.shape[1]))
        # xemb = xemb + posemb

        x = q + self.num_c * r
        qshftemb, xemb = self.exercise_emb(qry), self.interaction_emb(x)
    
        posemb = self.position_emb(pos_encode(xemb.shape[1]))
        xemb = xemb + posemb

        return qshftemb, xemb

    def forward(self, q, r, qry, qtest=False):
        emb_type = self.emb_type
        qemb, qshftemb, xemb = None, None, None
        if emb_type == "qid":
            qshftemb, xemb = self.base_emb(q, r, qry)
        # print(f"qemb: {qemb.shape}, xemb: {xemb.shape}, qshftemb: {qshftemb.shape}")
        for i in range(self.num_en):
            xemb = self.blocks[i](qshftemb, xemb, xemb)

        p = torch.sigmoid(self.pred(self.dropout_layer(xemb))).squeeze(-1)
        
        return p

class Blocks(Module):
    def __init__(self, emb_size, num_attn_heads, dropout) -> None:
        super().__init__()

        self.attn = MultiheadAttention(emb_size, num_attn_heads, dropout=dropout)
        self.attn_dropout = Dropout(dropout)
        self.attn_layer_norm = LayerNorm(emb_size)

        self.FFN = transformer_FFN(emb_size, dropout)
        self.FFN_dropout = Dropout(dropout)
        self.FFN_layer_norm = LayerNorm(emb_size)

    def forward(self, q=None, k=None, v=None):
        q, k, v = q.permute(1, 0, 2), k.permute(1, 0, 2), v.permute(1, 0, 2)
        # attn -> drop -> skip -> norm 
        # transformer: attn -> drop -> skip -> norm transformer default

        causal_mask = ut_mask(seq_len = k.shape[0])
        
        # YO TRIAL on causal_mask
        # nopeek_mask = np.triu(
        #     np.ones((k.shape[0], k.shape[0])), k=1).astype('uint8')
        # causal_mask = (torch.from_numpy(nopeek_mask) == 0).to(q.device)

        # import pdb; pdb.set_trace() 

        attn_emb, _ = self.attn(q, k, v, attn_mask=causal_mask)

        attn_emb = self.attn_dropout(attn_emb)
        attn_emb, q = attn_emb.permute(1, 0, 2), q.permute(1, 0, 2)

        attn_emb = self.attn_layer_norm(q + attn_emb)

        emb = self.FFN(attn_emb)
        emb = self.FFN_dropout(emb)
        emb = self.FFN_layer_norm(attn_emb + emb)
        return emb