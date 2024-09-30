import torch
from torch.nn import Module, Embedding, LSTM, Linear, Dropout
from torch.nn.functional import one_hot, binary_cross_entropy

from .qdkt import QueEmbedder
from .que_base_model import QueBaseModel

class DKTPlusQue(QueBaseModel):
    def __init__(self, num_c, emb_size, lambda_r, lambda_w1, lambda_w2, dropout=0.1, emb_type='qid', emb_path="", flag_load_emb=False, flag_emb_freezed=False, pretrain_dim=768,device='cpu',seed=0,mlp_layer_num=1,other_config={},**kwargs):
        model_name = "dkt_plus_que"

        super().__init__(model_name=model_name,emb_type=emb_type,emb_path=emb_path,pretrain_dim=pretrain_dim,device=device,seed=seed)
        self.model = DKTPlus(num_c=num_c,emb_size=emb_size,lambda_r=lambda_r, lambda_w1=lambda_w1, lambda_w2=lambda_w2, dropout=dropout,emb_type=emb_type, flag_load_emb=flag_load_emb, flag_emb_freezed=flag_emb_freezed,
                               emb_path=emb_path,pretrain_dim=pretrain_dim,device=device,mlp_layer_num=mlp_layer_num,other_config=other_config)
       
        self.model = self.model.to(device)
        self.emb_type = self.model.emb_type
        self.loss_func = self._get_loss_func("binary_crossentropy")
       
    def train_one_step(self,data,process=True,return_all=False, weighted_loss=0):
        outputs,data_new = self.predict_one_step(data,return_details=True,process=process)

        y_curr = torch.masked_select(outputs['y_curr'], data_new['sm'])
        y_next = torch.masked_select(outputs['y_next'], data_new['sm'])
        r_curr = torch.masked_select(data_new['r'], data_new['sm'])
        r_next = torch.masked_select(data_new['rshft'], data_new['sm'])
        # Get weights for prediction
        # weight_loss = get_class_weights(y_next, weighted_loss)
        # loss = binary_cross_entropy(y_next.float(), r_next.float(), weight=weight_loss)
        loss = self.get_loss(outputs['y_next'],data_new['rshft'],data_new['sm'], weighted_loss=weighted_loss)

        loss_r = binary_cross_entropy(y_curr.float(), r_curr.float()) # if answered wrong for C in t-1, cur answer for C should be wrong too
        loss_w1 = torch.masked_select(torch.norm(outputs['y'][:, 1:] - outputs['y'][:, :-1], p=1, dim=-1), data_new['sm'][:, 1:])
        loss_w1 = loss_w1.mean() / self.model.num_c
        loss_w2 = torch.masked_select(torch.norm(outputs['y'][:, 1:] - outputs['y'][:, :-1], p=2, dim=-1) ** 2, data_new['sm'][:, 1:])
        loss_w2 = loss_w2.mean() / self.model.num_c

        loss = loss + self.model.lambda_r * loss_r + self.model.lambda_w1 * loss_w1 + self.model.lambda_w2 * loss_w2

        return outputs['y_next'],loss#y_question没用

    def predict_one_step(self,data,return_details=False,process=True,return_raw=False):
        data_new = self.batch_to_device(data,process=process)

        y = self.model(data_new['q'].long(),data_new['r'].long())
        y_next = (y * one_hot(data_new['qshft'].long(), self.model.num_c)).sum(-1)
        y_curr = (y * one_hot(data_new['q'].long(), self.model.num_c)).sum(-1)

        outputs = {'y':y, 'y_next':y_next, 'y_curr':y_curr}

        if return_details:
            return outputs,data_new
        else:
            return outputs['y_next']

class DKTPlus(Module):
    def __init__(self, num_c, emb_size, lambda_r, lambda_w1, lambda_w2, dropout=0.1, emb_type="qid", emb_path="", flag_load_emb=False, flag_emb_freezed=False, pretrain_dim=768, **kwargs):
        super().__init__()
        self.model_name = "dkt_plus_que"
        self.num_c = num_c
        self.emb_size = emb_size
        self.hidden_size = emb_size
        self.lambda_r = lambda_r
        self.lambda_w1 = lambda_w1
        self.lambda_w2 = lambda_w2
        self.emb_type = emb_type

        if emb_type.startswith("qid"):
            self.interaction_emb = QueEmbedder(self.num_c, emb_size, emb_path, flag_load_emb, flag_emb_freezed, self.model_name)
        self.lstm_layer = LSTM(self.emb_size, self.hidden_size, batch_first=True)
        self.dropout_layer = Dropout(dropout)
        self.out_layer = Linear(self.hidden_size, self.num_c)
        

    def forward(self, q, r):
        emb_type = self.emb_type
        if emb_type == "qid":
            x = q + self.num_c * r
            xemb = self.interaction_emb(x)

        h, _ = self.lstm_layer(xemb)
        h = self.dropout_layer(h)
        y = self.out_layer(h)
        y = torch.sigmoid(y)

        return y