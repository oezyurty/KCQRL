import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from .akt_que import QueEmbedder as SkillEmbedder
from .qdkt import QueEmbedder as KnowledgeEmbedder
from .que_base_model import QueBaseModel

device = "cpu" if not torch.cuda.is_available() else "cuda"

class KQNQue(QueBaseModel):
    def __init__(self, n_skills:int, n_hidden:int, n_rnn_hidden:int, n_mlp_hidden:int, dropout, n_rnn_layers:int=1, rnn_type='lstm', emb_type="qid", emb_path="", flag_load_emb=False, flag_emb_freezed=False, pretrain_dim=768,  device='cpu',seed=0, **kwargs):
        model_name = "kqn_que"
        super().__init__(model_name=model_name,emb_type=emb_type,emb_path=emb_path,pretrain_dim=pretrain_dim,device=device,seed=seed)
        self.model = KQN(n_skills=n_skills, n_hidden=n_hidden, n_rnn_hidden=n_rnn_hidden, n_mlp_hidden=n_mlp_hidden, dropout=dropout, n_rnn_layers=n_rnn_layers, rnn_type=rnn_type, emb_type=emb_type, emb_path=emb_path, flag_load_emb=flag_load_emb, flag_emb_freezed=flag_emb_freezed, pretrain_dim=768, device=device, seed=seed)
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

class KQN(nn.Module):
    # n_skills: number of skills in dataset
    # n_hidden: dimensionality of skill and knowledge state vectors
    # n_rnn_hidden: number of hidden units in rnn knowledge encoder
    # n_mlp_hidden: number of hidden units in mlp skill encoder
    # n_rnn_layers: number of layers in rnn knowledge encoder
    # rnn_type: type of rnn cell, chosen from ['gru', 'lstm']
    def __init__(self, n_skills:int, n_hidden:int, n_rnn_hidden:int, n_mlp_hidden:int, dropout, n_rnn_layers:int=1, rnn_type='lstm', emb_type="qid", emb_path="", flag_load_emb=False, flag_emb_freezed=False, pretrain_dim=768, **kwargs):
        super(KQN, self).__init__()
        self.model_name = "kqn_que"
        self.emb_type = emb_type
        self.num_c = n_skills
        self.n_hidden = n_hidden
        self.n_rnn_hidden = n_rnn_hidden
        self.n_mlp_hidden = n_mlp_hidden
        self.n_rnn_layers = n_rnn_layers
        
        self.rnn_type, rnn_type = rnn_type.lower(), rnn_type.lower()

        if emb_type == "qid":
            if rnn_type == 'lstm':
                self.rnn = nn.LSTM(
                    input_size=n_hidden,
                    hidden_size=n_rnn_hidden,
                    num_layers=n_rnn_layers,
                    batch_first=True
                )
            elif rnn_type == 'gru':
                self.rnn = nn.GRU(
                    input_size=n_hidden,
                    hidden_size=n_rnn_hidden,
                    num_layers=n_rnn_layers,
                    batch_first=True
                )
        
        self.linear = nn.Linear(n_rnn_hidden, n_hidden)
        
        self.skill_encoder = nn.Sequential(
            nn.Linear(n_hidden, n_mlp_hidden),
            nn.ReLU(),
            nn.Linear(n_mlp_hidden, n_hidden),
            nn.ReLU()
        )
        self.drop_layer = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()
        # self.loss_fn = nn.BCEWithLogitsLoss(reduction='mean')

        self.embed_knowledge = KnowledgeEmbedder(self.num_c, n_hidden, emb_path, flag_load_emb, flag_emb_freezed, self.model_name)
        self.embed_skill = SkillEmbedder(self.num_c, n_hidden, emb_path, flag_load_emb, flag_emb_freezed, self.model_name)


    
    def init_hidden(self, batch_size: int):
        weight = next(self.parameters()).data
        if self.rnn_type == 'lstm':
            return (Variable(weight.new(self.n_rnn_layers, batch_size, self.n_rnn_hidden).zero_()),
                    Variable(weight.new(self.n_rnn_layers, batch_size, self.n_rnn_hidden).zero_()))
        else:
            return Variable(weight.new(self.n_rnn_layers, batch_size, self.n_rnn_hidden).zero_())
    
    
    def forward(self, q, r, qshft, qtest=False):
        # in_data = self.two_eye.to(q.device)[r * self.num_c + q]
        in_data = self.embed_knowledge(r * self.num_c + q)
        # next_skills = self.eye.to(q.device)[qshft]
        next_skills = self.embed_skill(qshft)
        # print(f"q: {q.tolist()}, r: {r.tolist()}")
        # print(f"in_data: {in_data.tolist()}")
        # import sys
        # sys.exit()
        emb_type = self.emb_type
        # print(f"in_data: {in_data.shape}")
        if emb_type == "qid":
            encoded_knowledge = self.encode_knowledge(in_data) # (batch_size, max_seq_len, n_hidden)
        encoded_skills = self.encode_skills(next_skills) # (batch_size, max_seq_len, n_hidden)
        encoded_knowledge = self.drop_layer(encoded_knowledge)
        
        # query the knowledge state with respect to the encoded skills
        # do the dot product
        logits = torch.sum(encoded_knowledge * encoded_skills, dim=2) # (batch_size, max_seq_len)
        logits = self.sigmoid(logits)
        if not qtest:
            return logits
        else:
            return logits, encoded_knowledge, encoded_skills

    def encode_knowledge(self, in_data):
        batch_size = in_data.size(0)
        self.hidden = self.init_hidden(batch_size)
        
        # rnn_input = pack_padded_sequence(in_data, seq_len, batch_first=True)
        rnn_output, _ = self.rnn(in_data, self.hidden)
        # rnn_output, _ = pad_packed_sequence(rnn_output, batch_first=True) # (batch_size, max_seq_len, n_rnn_hidden)
        encoded_knowledge = self.linear(rnn_output) # (batch_size, max_seq_len, n_hidden)
        return encoded_knowledge
    
    def encode_skills(self, next_skills):
        encoded_skills = self.skill_encoder(next_skills) # (batch_size, max_seq_len, n_hidden)
        encoded_skills = F.normalize(encoded_skills, p=2, dim=2) # L2-normalize
        return encoded_skills
