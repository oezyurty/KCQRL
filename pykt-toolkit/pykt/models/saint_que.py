import torch 
import torch.nn as nn
from torch.nn import Dropout
import pandas as pd
from .utils import transformer_FFN, get_clones, ut_mask, pos_encode
from torch.nn import Embedding, Linear

from .que_base_model import QueBaseModel
from .akt_que import QueEmbedder

device = "cpu" if not torch.cuda.is_available() else "cuda"

class SAINTQue(QueBaseModel):
    def __init__(self, num_q, num_c, seq_len, emb_size, num_attn_heads, dropout, n_blocks=1, emb_type="qid", emb_path="", flag_load_emb=False, flag_emb_freezed=False, pretrain_dim=768, device='cpu', seed=0, **kwargs):
        model_name = "saint_que"
        super().__init__(model_name=model_name,emb_type=emb_type,emb_path=emb_path,pretrain_dim=pretrain_dim,device=device,seed=seed)
        self.model = SAINT(num_q=num_q, num_c=num_c, seq_len=seq_len, 
            emb_size=emb_size, num_attn_heads=num_attn_heads, dropout=dropout, n_blocks=n_blocks, 
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
        y = self.model(data_new['cq'].long(), data_new['r'].long())
        outputs = {"y":y[:,1:]}
        if return_details:
            return outputs,data_new
        else:
            return outputs["y"]

class SAINT(nn.Module):
    def __init__(self, num_q, num_c, seq_len, emb_size, num_attn_heads, dropout, n_blocks=1, emb_type="qid", emb_path="", flag_load_emb=False, flag_emb_freezed=False, pretrain_dim=768, **kwargs):
        super().__init__()
        print(f"num_q: {num_q}, num_c: {num_c}")
        if num_q == num_c and num_q == 0:
            assert num_q != 0
        self.num_q = num_q
        self.num_c = num_c
        self.model_name = "saint_que"
        self.num_en = n_blocks
        self.num_de = n_blocks
        self.emb_type = emb_type

        self.embd_pos = nn.Embedding(seq_len, embedding_dim = emb_size) 
        # self.embd_pos = Parameter(torch.Tensor(seq_len-1, emb_size))
        # kaiming_normal_(self.embd_pos)

        if emb_type.startswith("qid"):
            #self.encoder = get_clones(Encoder_block(emb_size, num_attn_heads, num_q, num_c, seq_len, dropout), self.num_en)
            self.encoder = QueEmbedder(num_q, emb_size, emb_path, flag_load_emb, flag_emb_freezed, self.model_name)

        # self.layer_norm = nn.LayerNorm(emb_size)
        
        self.decoder = get_clones(Decoder_block(emb_size, 2, num_attn_heads, seq_len, dropout), self.num_de)

        self.dropout = Dropout(dropout)
        self.out = nn.Linear(in_features=emb_size, out_features=1)
    
    def forward(self, cq, r):
        emb_type = self.emb_type        

        in_pos = pos_encode(cq.shape[1])
        in_pos = self.embd_pos(in_pos)
        # in_pos = self.embd_pos.unsqueeze(0)
        ## pass through each of the encoder blocks in sequence

        emb_ex = self.encoder(cq) + in_pos
        # emb_ex = self.layer_norm(emb_ex)
        ## pass through each decoder blocks in sequence
        start_token = torch.tensor([[2]]).repeat(r.shape[0], 1).to(device)
        in_res = torch.cat((start_token, r), dim=-1)

        first_block = True
        for i in range(self.num_de):
            if i >= 1:
                first_block = False
            in_res = self.decoder[i](in_res, in_pos, en_out=emb_ex, first_block=first_block)
        
        ## Output layer

        res = self.out(self.dropout(in_res))
        res = torch.sigmoid(res).squeeze(-1)
        
        return res


class Decoder_block(nn.Module):
    """
    M1 = SkipConct(Multihead(LayerNorm(Qin;Kin;Vin)))
    M2 = SkipConct(Multihead(LayerNorm(M1;O;O)))
    L = SkipConct(FFN(LayerNorm(M2)))
    """

    def __init__(self, dim_model, total_res, heads_de, seq_len, dropout):
        super().__init__()
        self.seq_len    = seq_len
        self.embd_res    = nn.Embedding(total_res+1, embedding_dim = dim_model)                  #response embedding, include a start token
        # self.embd_pos   = nn.Embedding(seq_len, embedding_dim = dim_model)                  #positional embedding
        self.multi_de1  = nn.MultiheadAttention(embed_dim= dim_model, num_heads= heads_de, dropout=dropout)  # M1 multihead for interaction embedding as q k v
        self.multi_de2  = nn.MultiheadAttention(embed_dim= dim_model, num_heads= heads_de, dropout=dropout)  # M2 multihead for M1 out, encoder out, encoder out as q k v
        self.ffn_en     = transformer_FFN(dim_model, dropout)                                         # feed forward layer

        self.layer_norm1 = nn.LayerNorm(dim_model)
        self.layer_norm2 = nn.LayerNorm(dim_model)
        self.layer_norm3 = nn.LayerNorm(dim_model)

        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)


    def forward(self, in_res, in_pos, en_out,first_block=True):

         ## todo create a positional encoding (two options numeric, sine)
        if first_block:
            in_in = self.embd_res(in_res)

            #combining the embedings
            out = in_in + in_pos                         # (b,n,d)
        else:
            out = in_res

        # in_pos = get_pos(self.seq_len)
        # in_pos = self.embd_pos(in_pos)

        out = out.permute(1,0,2)                                    # (n,b,d)# print('pre multi', out.shape)
        n,_,_ = out.shape

        #Multihead attention M1                                     ## todo verify if E to passed as q,k,v
        out = self.layer_norm1(out)
        skip_out = out
        out, attn_wt = self.multi_de1(out, out, out, 
                                     attn_mask=ut_mask(seq_len=n)) # attention mask upper triangular
        out = self.dropout1(out)
        out = skip_out + out                                        # skip connection

        #Multihead attention M2                                     ## todo verify if E to passed as q,k,v
        en_out = en_out.permute(1,0,2)                              # (b,n,d)-->(n,b,d)
        en_out = self.layer_norm2(en_out)
        skip_out = out
        out, attn_wt = self.multi_de2(out, en_out, en_out,
                                    attn_mask=ut_mask(seq_len=n))  # attention mask upper triangular
        out = self.dropout2(out)
        out = out + skip_out

        #feed forward
        out = out.permute(1,0,2)                                    # (b,n,d)
        out = self.layer_norm3(out)                               # Layer norm 
        skip_out = out
        out = self.ffn_en(out)                                    
        out = self.dropout3(out)
        out = out + skip_out                                        # skip connection

        return out