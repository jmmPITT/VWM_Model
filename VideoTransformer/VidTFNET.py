
import os
import torch
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torch.nn.utils.rnn as rnn_utils

import torch
import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self, beta, input_dims=32, n_actions=4):
        super(TransformerBlock, self).__init__()
        self.input_dims = 128 + 4 + 8
        self.seq_length = 8 * 4
        self.n_actions = n_actions
        self.d_model = self.input_dims
        self.num_heads = 1
        self.dff = 1024
        self.dropout = 0.01

        self.d_model = self.d_model
        self.num_heads = self.num_heads
        self.d_k = self.d_model // self.num_heads

        ### query, key, and value weights ###
        self.W_q = nn.Linear(self.d_model, self.d_k*self.num_heads)
        self.W_k = nn.Linear(self.d_model, self.d_k*self.num_heads)
        self.W_v = nn.Linear(self.d_model, self.d_k*self.num_heads)
        # self.fc = nn.Linear(self.d_model, self.d_k)

        ### Position wise feed forward layers ###
        self.linear1 = nn.Linear(self.num_heads * self.d_k, self.dff)
        self.linear2 = nn.Linear(self.dff, self.d_model)
        self.dropout1 = nn.Dropout(self.dropout)

        ### norms and second dropout layer ###
        self.norm1 = nn.LayerNorm(self.d_model)
        self.norm2 = nn.LayerNorm(self.d_model)
        self.dropout2 = nn.Dropout(self.dropout)



        self.optimizer = optim.Adam(self.parameters(), lr=beta, betas=(0.9, 0.99), eps=1e-8, weight_decay=1e-7)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state, m):

        state = state.view(-1, self.seq_length, self.input_dims)

        batch_size = state.shape[0]

        ### mask values to 0 ###
        state = state.view(batch_size, self.seq_length, self.input_dims)

        ### Construct query, key, and value matrices for each head ###
        ### split among heads ###
        q = self.W_q(state).view(batch_size, -1, self.num_heads, self.d_k)
        k = self.W_k(state).view(batch_size, -1, self.num_heads, self.d_k)
        v = self.W_v(state).view(batch_size, -1, self.num_heads, self.d_k)
        q, k, v = [x.transpose(1, 2) for x in (q, k, v)]  # [batch_size, num_heads, seq_len, d_k]

        ### compute attention*values ###
        attn_values, A = self.calculate_attention(q, k, v, m)
        attn_values = attn_values.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)

        ### Feedforward and layer norm layers ###
        Z1 = state + self.dropout1(attn_values)
        Z2 = self.norm1(Z1)
        Z3 = F.relu(self.linear1(Z2))
        Z4 = self.linear2(Z3)
        Z5 = Z2 + self.dropout2(Z4)
        src = self.norm2(Z5)

        return src, A

    def calculate_attention(self, q, k, v, mask):
        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))

        mask = mask.unsqueeze(1).unsqueeze(2)

        if mask is not None:
            scores = scores.masked_fill(mask == True, float('-inf'))

        A = F.softmax(scores, dim=-1)


        return torch.matmul(A, v), A

class TransformerNetwork(nn.Module):
    def __init__(self, beta, input_dims=32, hidden_dim=128, fc1_dims=64, fc2_dims=32, n_actions=4,
                 name='transformer', chkpt_dir='td3_MAT'):
        super(TransformerNetwork, self).__init__()
        self.input_dims = 128 + 4 + 8
        self.hidden_dim = hidden_dim
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.seq_length = 8 * 4
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = name + '_td3'
        self.d_model = self.input_dims
        self.num_heads = 1
        self.dff = 1024
        self.dropout = 0.01

        # Embedding Generator
        self.fc_state_projection = nn.Linear(self.input_dims, self.input_dims)
        self.ln_state_projection = nn.LayerNorm(self.input_dims)

        self.fc2_state_projection = nn.Linear(self.input_dims, self.input_dims)
        self.ln2_state_projection = nn.LayerNorm(self.input_dims)


        self.transformer_block1 = TransformerBlock(beta, input_dims=self.input_dims, hidden_dim=self.hidden_dim, fc1_dims=self.fc1_dims, fc2_dims=self.fc2_dims, n_actions=self.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=beta, betas=(0.9, 0.99), eps=1e-8, weight_decay=1e-7)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)


    def forward(self, state, t, m):

        state = state.view(-1, self.seq_length, self.input_dims)

        batch_size = state.shape[0]

        state = state.view(batch_size * self.seq_length, self.input_dims)

        ### Generate Embedding ###
        X = F.elu(state + self.ln_state_projection(self.fc_state_projection(state)))
        state = self.ln2_state_projection(state + self.fc2_state_projection(X))


        # Adding Gaussian noise
        std = 0.001  # Adjust this value as needed
        noise = torch.randn_like(state) * std
        state = state + noise

        ### mask values to 0 ###
        state = state.view(batch_size, self.seq_length, self.input_dims)
        state = state * m.view(batch_size, self.seq_length, self.input_dims)
        mask = (state == 0).all(dim=-1)  # Creates a mask of shape [batch_size, seq_len]

        src,A1 = self.transformer_block1(state, mask)

        return src

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.checkpoint_file))


class CriticNetwork(nn.Module):
    def __init__(self, beta, input_dims=32, hidden_dim=512, fc1_dims=256, fc2_dims=128, n_actions=4,
                 name='critic', chkpt_dir='td3_MAT'):
        super(CriticNetwork, self).__init__()
        self.input_dims = 128 + 4 + 8
        self.hidden_dim = hidden_dim
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.seq_length = 8 * 4
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = name + '_td3'

        self.fc_action_lever_projection = nn.Linear(self.n_actions, self.input_dims * self.seq_length)
        self.ln_action_lever_projection = nn.LayerNorm(self.input_dims * self.seq_length)


        self.fc1 = nn.Linear(self.input_dims * self.seq_length + self.input_dims * self.seq_length, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)

        # Second hidden layer and batch norm
        self.fc2 = nn.Linear(hidden_dim, fc1_dims)
        self.ln2 = nn.LayerNorm(fc1_dims)

        # Output layer - representing Q-values for each action
        self.fc3 = nn.Linear(fc1_dims, fc2_dims)
        self.ln3 = nn.LayerNorm(fc2_dims)
        self.q = nn.Linear(fc2_dims, 2)  # policy action selection

        self.optimizer = optim.Adam(self.parameters(), lr=beta, betas=(0.9, 0.99), eps=1e-8, weight_decay=1e-6)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, transformer_state, action, obs):
        transformer_state = transformer_state.view(-1, self.input_dims * self.seq_length)


        # Concatenating along the second dimension (dim=1)
        action = action.view(-1, self.n_actions)

        action_lever = self.fc_action_lever_projection(action)

        state_action = torch.cat((transformer_state, action_lever), dim=1)

        x = F.elu(self.ln1(self.fc1(state_action)))
        x = F.elu(self.ln2(self.fc2(x)))
        x = F.elu(self.ln3(self.fc3(x)))

        q = self.q(x)
        q = F.softmax(q, dim=1)
        return q

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.checkpoint_file))


class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims=16, hidden_dim=512, fc1_dims=256, fc2_dims=128, n_actions=4, name='Actor',
                 chkpt_dir='td3_MAT'):
        super(ActorNetwork, self).__init__()
        self.input_dims = 128 + 4 + 8
        self.hidden_dim = hidden_dim
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.seq_length = 8 * 4
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = name + '_td3'

        self.fc1 = nn.Linear(self.input_dims * self.seq_length, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)

        # Second hidden layer and batch norm
        self.fc2 = nn.Linear(hidden_dim, fc1_dims)
        self.ln2 = nn.LayerNorm(fc1_dims)

        # Output layer - representing Q-values for each action
        self.fc3 = nn.Linear(fc1_dims, fc2_dims)
        self.ln3 = nn.LayerNorm(fc2_dims)
        self.pi_lever = nn.Linear(fc2_dims, self.n_actions)  # policy action selection


        self.optimizer = optim.Adam(self.parameters(), lr=alpha, betas=(0.9, 0.99), eps=1e-8, weight_decay=1e-6)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, transformer_state, obs):

        transformer_state = transformer_state.view(-1, self.input_dims * self.seq_length)

        x = F.elu(self.ln1(self.fc1(transformer_state)))
        x = F.elu(self.ln2(self.fc2(x)))
        x = F.elu(self.ln3(self.fc3(x)))


        # lever action
        pi_lever = F.softmax(self.pi_lever(x),dim=-1)

        return pi_lever

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.checkpoint_file))