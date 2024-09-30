import torch

from torch import nn
from torch.nn import Module, Embedding, LSTM, Linear, Dropout, LayerNorm, TransformerEncoder, TransformerEncoderLayer, CrossEntropyLoss
from .utils import ut_mask

from torch.nn.functional import one_hot, binary_cross_entropy

from .que_base_model import QueBaseModel
from .qdkt import QueEmbedder

device = "cpu" if not torch.cuda.is_available() else "cuda"

class ATDKTQue(QueBaseModel):
    def __init__(self, num_q,num_c, emb_size, dropout=0.1, emb_type='qid', loss_pred=0.5, loss_his=0.5, start=50, emb_path="", flag_load_emb=False, flag_emb_freezed=False, pretrain_dim=768,device='cpu',seed=0, other_config={},**kwargs):
        model_name = "atdkt_que"

        super().__init__(model_name=model_name,emb_type=emb_type,emb_path=emb_path,pretrain_dim=pretrain_dim,device=device,seed=seed)
        self.model = ATDKT(num_q=num_q,num_c=num_c,emb_size=emb_size,dropout=dropout,emb_type=emb_type, loss_pred=loss_pred, loss_his=loss_his, start=start, flag_load_emb=flag_load_emb, flag_emb_freezed=flag_emb_freezed,
                               emb_path=emb_path,pretrain_dim=pretrain_dim,device=device, other_config=other_config)
       
        self.model = self.model.to(device)
        self.emb_type = self.model.emb_type
        self.loss_func = self._get_loss_func("binary_crossentropy")
       
    def train_one_step(self,data,process=True,return_all=False, weighted_loss=0):
        outputs,data_new = self.predict_one_step(data,return_details=True,process=process, train=True)

        loss_pred = self.get_loss(outputs['y'],data_new['rshft'],data_new['sm'], weighted_loss=weighted_loss)
        loss = self.model.loss_pred * loss_pred + self.model.loss_his * outputs["loss_his"]

        return outputs['y'],loss#y_question没用

    def predict_one_step(self,data,return_details=False,process=True,return_raw=False, train=False):
        data_new = self.batch_to_device(data,process=process)
        outputs = self.model(data_new, train=train)
        if return_details:
            return outputs,data_new
        else:
            return outputs['y']

class ATDKT(Module):
    def __init__(self, num_q, num_c, emb_size, dropout=0.1, emb_type='qid', 
            loss_pred=0.5, loss_his=0.5, l3=0.5, start=50, emb_path="", flag_load_emb=False, flag_emb_freezed=False, pretrain_dim=768, **kwargs):
        super().__init__()
        self.model_name = "atdkt_que"
        print(f"qnum: {num_q}, cnum: {num_c}")
        print(f"emb_type: {emb_type}")
        self.num_q = num_q
        self.num_c = num_c
        self.emb_size = emb_size
        self.hidden_size = emb_size
        self.emb_type = emb_type

        #self.interaction_emb = Embedding(self.num_c * 2, self.emb_size)
        self.interaction_emb = QueEmbedder(self.num_q, emb_size, emb_path, flag_load_emb, flag_emb_freezed, self.model_name)

        self.lstm_layer = LSTM(self.emb_size, self.hidden_size, batch_first=True)
        self.dropout_layer = Dropout(dropout)
        self.out_layer = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size//2), nn.ReLU(), nn.Dropout(dropout),
                Linear(self.hidden_size//2, self.num_q))

        self.loss_pred = loss_pred
        self.loss_his = loss_his
        
        self.start = start
        self.hisclasifier = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size//2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(self.hidden_size//2, 1))
        self.hisloss = nn.MSELoss()

    def get_historycorrs(self, r):
        # Step 1: Compute cumulative sums along the sequence dimension
        cumulative_sums = torch.cumsum(r, dim=1)

        # Step 2: Create an array for the sequence 1, 2, ..., sequence_length
        denominators = torch.arange(1, r.shape[1] + 1).unsqueeze(0).type_as(r)

        # Step 3: Divide cumulative sums by the denominators
        historycorrs = cumulative_sums / denominators

        return historycorrs

    def forward(self, dcur, train=False): ## F * xemb
        # print(f"keys: {dcur.keys()}")
        q, r = dcur["q"].long(), dcur["r"].long()

        historycorrs = self.get_historycorrs(r)
        
        y2, y3 = 0, 0

        emb_type = self.emb_type

        x = q + self.num_q * r
        xemb = self.interaction_emb(x)
        rpreds, qh = None, None

        # predict response
        h, _ = self.lstm_layer(xemb)
        # predict history correctness rates
        if train:
            sm = dcur["sm"].long()
            start = self.start
            rpreds = torch.sigmoid(self.hisclasifier(h)[:,start:,:]).squeeze(-1)
            rsm = sm[:,start:]
            rflag = rsm==1
            rtrues = historycorrs[:,start:]
            loss_his = self.hisloss(rpreds[rflag], rtrues[rflag])

        h = self.dropout_layer(h)
        y = self.out_layer(h)
        y = torch.sigmoid(y)

        y = (y * one_hot(dcur['qshft'].long(), self.num_q)).sum(-1)

        outputs = {"y":y}
        if train:
            outputs["loss_his"] = loss_his

        return outputs

