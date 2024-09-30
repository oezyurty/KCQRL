## YO Edition:
## Implementation of Causal Simple KT (CSKT) model  

import torch
import torch.nn as nn
import torch.nn.functional as F
import json

class CausalSimpleKT(nn.Module):
    def __init__(self, num_kcs, kc_dim, state_dim, dropout_rate=0.1, emb_type="qid", question_embedding_path=''):
        super(CausalSimpleKT, self).__init__()
        self.model_name = "cskt"
        self.emb_type = emb_type
        # Load pre-computed question embeddings
        with open(question_embedding_path, 'r') as f:
            precomputed_embeddings = json.load(f)
        precomputed_embeddings_tensor = torch.tensor([precomputed_embeddings[str(i)] for i in range(len(precomputed_embeddings))], dtype=torch.float)
        
        # Define the dimensions
        self.state_dim = state_dim
        self.kc_dim = kc_dim
        self.question_dim = precomputed_embeddings_tensor.size(1)
        self.response_emb_dim = kc_dim
        self.prediction_dim = 1
        self.num_kcs = num_kcs
        # Define a learnable initial state for the model
        # The initial state is defined as a parameter, which allows it to be optimized during training
        # It's initialized to a tensor of zeros
        self.initial_state = nn.Parameter(torch.zeros(1, state_dim))

        # Embedding layers
        self.question_embeddings = nn.Embedding.from_pretrained(precomputed_embeddings_tensor, freeze=True)
        self.kc_embeddings = nn.Embedding(num_kcs, kc_dim)
        self.response_embeddings = nn.Embedding(2, kc_dim)

        # Define layers for computing Response based on State and KC
        self.fc_state_kc = nn.Linear(state_dim + kc_dim, self.prediction_dim)
        #self.dropout_state_kc = nn.Dropout(dropout_rate)
        #self.layer_norm_state_kc = nn.LayerNorm(response_dim)

        # Define layers for computing Response based on State and Question
        self.fc_state_question = nn.Linear(state_dim + self.question_dim, self.prediction_dim)
        #self.dropout_state_question = nn.Dropout(dropout_rate)
        #self.layer_norm_state_question = nn.LayerNorm(response_dim)

        # Define layers for updating State
        self.fc_state_update = nn.Linear(state_dim + kc_dim + self.question_dim + self.response_emb_dim, state_dim)
        self.dropout_state_update = nn.Dropout(dropout_rate)
        self.layer_norm_state_update = nn.LayerNorm(state_dim)
        
        # Learnable mixing parameter, initialized to a value that results in 0.5 after applying the sigmoid function
        # The inverse sigmoid of 0.5 is 0 (logit(0.5) = 0), so we initialize it to 0
        self.alpha_logit = nn.Parameter(torch.tensor([0.0]))

    def forward(self, question_ids, kc_ids, responses):
        # Convert KC and Question IDs to embeddings
        kc_embeddings = self.kc_embeddings(kc_ids)  # Assuming kc_ids is [batch_size, seq_len]
        question_embeddings = self.question_embeddings(question_ids)  # Assuming question_ids is [batch_size, seq_len]
        response_embeddings = self.response_embeddings(responses)  # Assuming responses is [batch_size, seq_len]
        
        # Assume kc, question, and initial_state are properly batched and have dimensions [batch_size, seq_len, feature_dim]
        batch_size, seq_len = kc_ids.size()
        hidden_states = []
        predicted_responses = []

        # Replicate the initial state across the batch dimension
        state = self.initial_state.expand(batch_size, -1)

        for i in range(seq_len):
            kc_i = kc_embeddings[:, i, :]
            question_i = question_embeddings[:, i, :]
            response_i = response_embeddings[:, i, :]

            # Compute Response based on State and KC
            response_from_kc = self.fc_state_kc(torch.cat((state, kc_i), dim=1))
            response_from_kc = torch.sigmoid(response_from_kc)

            # Compute Response based on State and Question
            response_from_question = self.fc_state_question(torch.cat((state, question_i), dim=1))
            response_from_question = torch.sigmoid(response_from_question)

            # Apply sigmoid to alpha_logit to ensure it's between 0 and 1
            alpha = torch.sigmoid(self.alpha_logit)
            # Combine responses using the learnable alpha
            response = alpha * response_from_kc + (1 - alpha) * response_from_question

            # Update State
            state_update = self.fc_state_update(torch.cat((state, kc_i, question_i, response_i), dim=1))
            state_update = self.dropout_state_update(state_update)
            state = torch.tanh(self.layer_norm_state_update(state_update))

            hidden_states.append(state.unsqueeze(1))
            predicted_responses.append(response.unsqueeze(1))

        # Concatenate states and responses over the sequence length dimension
        hidden_states = torch.cat(hidden_states, dim=1)
        predicted_responses = torch.cat(predicted_responses, dim=1)
        
        # Squeeze the last dimension of the predicted_responses tensor
        # Assuming the dimensions are [batch_size, seq_len, 1]
        predicted_responses = predicted_responses.squeeze(-1)

        return predicted_responses, hidden_states