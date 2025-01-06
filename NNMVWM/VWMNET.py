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


class CustomLSTMCell(nn.Module):
    def __init__(self, patch_size, d_model):
        super(CustomLSTMCell, self).__init__()
        self.patch_size = patch_size
        self.d_model = d_model
        self.dff = 1024
        self.input_size = self.d_model
        self.patch_size = patch_size

        # Combine input and hidden state transformations for each gate into linear transformations
        self.WI = nn.Linear(self.dff, self.dff)
        self.WF = nn.Linear(self.dff, self.dff)
        self.WO = nn.Linear(self.dff, self.dff)
        self.WZ = nn.Linear(self.dff, self.dff)

        self.RI = nn.Linear(self.input_size, self.dff)
        self.RF = nn.Linear(self.input_size, self.dff)
        self.RO = nn.Linear(self.input_size, self.dff)
        self.RZ = nn.Linear(self.input_size, self.dff)


    def forward(self, Zi, Ci, Mi, Hi, Ni):
        m = m.view(-1, self.patch_size, 1)

        Zi = Zi.view(-1, self.patch_size, self.input_size)
        Ci = Ci.view(-1, self.patch_size, self.dff)
        Hi = Hi.view(-1, self.patch_size, self.dff)
        Ni = Ni.view(-1, self.patch_size, self.dff)

        C_prev = Ci
        M_prev = Mi
        H_prev = Hi
        N_prev = Ni


        I_tilde = self.WI(H_prev) + self.RI(Zi)
        # I_t = torch.exp(I_tilde)

        F_tilde = self.WF(H_prev) + self.RF(Zi)
        # F_t = torch.exp(F_tilde)

        O_tilde = self.WO(H_prev) + self.RO(Zi)

        Z_tilde = self.WZ(H_prev) + self.RZ(Zi)



        M_t = torch.max(F_tilde + M_prev, I_tilde)
        I_t = torch.exp(I_tilde - M_t)
        F_t = torch.exp(F_tilde + M_prev - M_t)


        O_t = F.sigmoid(O_tilde)
        N_t = F_t*N_prev + I_t
        Z_t = F.tanh(Z_tilde)
        C_t = (C_prev * F_t + Z_t * I_t)  + C_prev
        H_t = O_t * (C_t/N_t) * (m) + H_prev




        return C_t, M_t, H_t, N_t

class TransformerBlock(nn.Module):
    def __init__(self, beta, input_dims=32, hidden_dim=128, fc1_dims=64, fc2_dims=32, n_actions=4,
                 name='transformer', chkpt_dir='td3_MAT'):
        super(TransformerBlock, self).__init__()
        self.input_dims = (128 + 4 + 8)
        self.hidden_dim = hidden_dim
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.patch_length = 4
        self.seq_length = 8
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

        self.W_Cq = nn.Linear(self.dff, self.d_k * self.num_heads)
        self.W_Ck = nn.Linear(self.dff, self.d_k * self.num_heads)
        self.W_Cv = nn.Linear(self.dff, self.d_k * self.num_heads)

        ### Position wise feed forward layers ###
        self.linear1 = nn.Linear(self.num_heads * self.d_k, self.dff)
        self.linear2 = nn.Linear(self.dff, self.d_model)
        self.dropout1 = nn.Dropout(self.dropout*2)

        ### norms and second dropout layer ###
        self.norm1 = nn.LayerNorm(self.d_model)
        self.norm2 = nn.LayerNorm(self.d_model)
        self.dropout2 = nn.Dropout(self.dropout)


        self.LSTM = CustomLSTMCell(patch_size=self.patch_length, d_model=self.d_model)




        self.optimizer = optim.Adam(self.parameters(), lr=beta, betas=(0.9, 0.99), eps=1e-8, weight_decay=1e-6)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state, m):

        state = state.view(-1, self.seq_length, self.patch_length, self.input_dims)
        m = m.view(-1, self.seq_length, self.patch_length, 1)

        batch_size = state.shape[0]

        C = torch.zeros(batch_size, self.patch_length, self.dff).to(self.device)
        M = torch.zeros(batch_size, self.patch_length, self.dff).to(self.device)
        H = torch.zeros(batch_size, self.patch_length, self.dff).to(self.device)
        N = torch.zeros(batch_size, self.patch_length, self.dff).to(self.device)

        # Initialize an empty list to collect the tensors
        A_list = []

        for i in range(self.seq_length):

            ### mask values to 0 ###
            state_i = state[:, i, :, :].view(batch_size, self.patch_length, self.input_dims)

            ### Construct query, key, and value matrices for each head ###
            ### split among heads ###
            ### Construct query, key, and value matrices for each head ###
            ### split among heads ###
            q = self.W_q(state_i).view(batch_size, -1, self.num_heads, self.d_k) * self.W_Cq(H).view(batch_size, -1, self.num_heads, self.d_k)
            k = self.W_k(state_i).view(batch_size, -1, self.num_heads, self.d_k) * self.W_Ck(H).view(batch_size, -1, self.num_heads, self.d_k)
            v = self.W_v(state_i).view(batch_size, -1, self.num_heads, self.d_k) * self.W_Cv(H).view(batch_size, -1, self.num_heads, self.d_k)
            q, k, v = [x.transpose(1, 2) for x in (q, k, v)]  # [batch_size, num_heads, seq_len, d_k]

            ### compute attention*values ###
            attn_values, A = self.calculate_attention(q, k, v, m)
            attn_values = attn_values.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)

            ### Feedforward and layer norm layers ###
            Z1 = state_i + self.dropout1(attn_values)
            Z2 = self.norm1(Z1)

            m_i = m[:,i,:,:].view(-1, self.patch_length, 1)

            C, M, H, N = self.LSTM(Z2, C, M, H, N, m_i)

            A_list.append(A)

        # Concatenate all tensors in the list along a new dimension (1) to get [batch, seq_length, patch_length, input_dims]
        A_tensor = torch.stack(A_list, dim=1)

        return H, A_tensor

    def calculate_attention(self, q, k, v, mask):
        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))
        A = F.softmax(scores, dim=-1)
        return torch.matmul(A, v), A

class TransformerNetwork(nn.Module):
    def __init__(self, beta, input_dims=32, hidden_dim=128, fc1_dims=64, fc2_dims=32, n_actions=4,
                 name='transformer', chkpt_dir='td3_MAT'):
        super(TransformerNetwork, self).__init__()
        self.input_dims = (128 + 4 + 8)
        self.hidden_dim = hidden_dim
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.patch_length = 4
        self.seq_length = 8
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = name + '_td3'
        self.d_model = self.input_dims
        self.num_heads = 1
        self.dff = 1024
        self.dropout = 0.01
        # self.embed_dim = 128

        # Embedding Generator
        self.fc_state_projection = nn.Linear(self.input_dims, self.input_dims)
        self.ln_state_projection = nn.LayerNorm(self.input_dims)

        self.fc2_state_projection = nn.Linear(self.input_dims, self.input_dims)
        self.ln2_state_projection = nn.LayerNorm(self.input_dims)

        # self.fc3_state_projection = nn.Linear(self.input_dims, self.input_dims)

        # GRU Layer
        # self.LSTM = nn.LSTM(input_size=input_dims*2, hidden_size=self.input_dims, num_layers=1, batch_first=True)

        self.transformer_block1 = TransformerBlock(beta, input_dims=128, hidden_dim=self.hidden_dim, fc1_dims=self.fc1_dims, fc2_dims=self.fc2_dims, n_actions=self.n_actions)
        # self.transformer_block2 = TransformerBlock(beta, input_dims=128, hidden_dim=self.hidden_dim, fc1_dims=self.fc1_dims, fc2_dims=self.fc2_dims, n_actions=self.n_actions)
        # self.transformer_block3 = TransformerBlock(beta, input_dims=self.input_dims, hidden_dim=self.hidden_dim, fc1_dims=self.fc1_dims, fc2_dims=self.fc2_dims, n_actions=self.n_actions)

        ### cue predictor ###
        # self.cue_predictor = nn.Linear(self.input_dims*self.seq_length, 4)
        #
        # ### next action predictor ###
        # self.action_predictor = nn.Linear(self.input_dims*self.seq_length, 3)
        #
        # ### reward predictor ###
        # self.reward_predictor = nn.Linear(self.input_dims * self.seq_length, 1)

        ### target predictor ###
        # self.fc1_target_predictor = nn.Linear(self.input_dims * self.seq_length, self.fc1_dims)
        # self.ln1_target_predictor = nn.LayerNorm(self.fc1_dims)
        # self.fc2_target_predictor = nn.Linear(self.fc1_dims, 5)
        # self.ln2_target_predictor = nn.LayerNorm(self.fc2_dims)

        self.optimizer = optim.Adam(self.parameters(), lr=beta, betas=(0.9, 0.99), eps=1e-8, weight_decay=1e-6)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)


    def forward(self, state):
        # print('STATE!!!', state.shape)
        state = state.view(-1, self.seq_length, self.patch_length, self.input_dims)

        batch_size = state.shape[0]

        state = state.view(batch_size * self.seq_length * self.patch_length, self.input_dims)

        ### Generate Embedding ###
        X = F.elu(state + self.ln_state_projection(self.fc_state_projection(state)))
        state = self.ln2_state_projection(state + self.fc2_state_projection(X))

        # Adding Gaussian noise
        std = 0.001  # Adjust this value as needed
        noise = torch.randn_like(state) * std
        state = state + noise

        state = state.view(batch_size, self.seq_length, self.patch_length, self.input_dims)

        src,A1 = self.transformer_block1(state)

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
        self.input_dims = 1024
        self.hidden_dim = hidden_dim
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.patch_length = 4
        self.seq_length = 1

        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = name + '_td3'

        self.fc_action_lever_projection = nn.Linear(self.n_actions, self.input_dims * self.patch_length*self.seq_length)
        self.ln_action_lever_projection = nn.LayerNorm(self.input_dims * self.patch_length*self.seq_length)
        #
        # self.fc_action_sensor_projection = nn.Linear(16, self.input_dims * 2)
        # self.ln_action_sensor_projection = nn.LayerNorm(self.input_dims * 2)

        self.fc1 = nn.Linear(self.input_dims * self.patch_length*self.seq_length + self.input_dims * self.patch_length*self.seq_length, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        # self.dropout1 = nn.Dropout(dropout_rate)

        # Second hidden layer and batch norm
        self.fc2 = nn.Linear(hidden_dim, fc1_dims)
        self.ln2 = nn.LayerNorm(fc1_dims)
        # self.dropout2 = nn.Dropout(dropout_rate)

        # Output layer - representing Q-values for each action
        self.fc3 = nn.Linear(fc1_dims, fc2_dims)
        self.ln3 = nn.LayerNorm(fc2_dims)
        # self.dropout3 = nn.Dropout(dropout_rate)
        self.q = nn.Linear(fc2_dims, 10)  # policy action selection

        self.optimizer = optim.Adam(self.parameters(), lr=beta, betas=(0.9, 0.99), eps=1e-8, weight_decay=1e-6)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, transformer_state, action, obs):
        transformer_state = transformer_state.view(-1, self.input_dims * self.patch_length*self.seq_length)

        # print('TRANSFORM!!!!',transformer_state.shape)

        # Concatenating along the second dimension (dim=1)
        action = action.view(-1, self.n_actions)
        # print('action!!!!',action.shape)

        action_lever = self.fc_action_lever_projection(action)
        # print('action_lever!!!!',action_lever.shape)

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
        self.input_dims = 1024
        self.hidden_dim = hidden_dim
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.patch_length = 4
        self.seq_length = 1
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = name + '_td3'

        self.fc1 = nn.Linear(self.input_dims * self.patch_length*self.seq_length, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        # self.dropout1 = nn.Dropout(dropout_rate)

        # Second hidden layer and batch norm
        self.fc2 = nn.Linear(hidden_dim, fc1_dims)
        self.ln2 = nn.LayerNorm(fc1_dims)
        # self.dropout2 = nn.Dropout(dropout_rate)

        # Output layer - representing Q-values for each action
        self.fc3 = nn.Linear(fc1_dims, fc2_dims)
        self.ln3 = nn.LayerNorm(fc2_dims)
        # self.dropout3 = nn.Dropout(dropout_rate)
        self.pi_lever = nn.Linear(fc2_dims, self.n_actions)  # policy action selection

        # self.fc1_alpha = nn.Linear(self.input_dims*2*7, hidden_dim)
        # self.ln1_alpha = nn.LayerNorm(hidden_dim)
        # # self.dropout1 = nn.Dropout(dropout_rate)
        #
        # # Second hidden layer and batch norm
        # self.fc2_alpha = nn.Linear(hidden_dim, fc1_dims)
        # self.ln2_alpha = nn.LayerNorm(fc1_dims)
        # # self.dropout2 = nn.Dropout(dropout_rate)
        #
        # # Output layer - representing Q-values for each action
        # self.fc3_alpha = nn.Linear(fc1_dims, fc2_dims)
        # self.ln3_alpha = nn.LayerNorm(fc2_dims)
        # # self.dropout3 = nn.Dropout(dropout_rate)
        # self.pi_sensor = nn.Linear(fc2_dims, 16)  # policy action selection

        self.optimizer = optim.Adam(self.parameters(), lr=alpha, betas=(0.9, 0.99), eps=1e-8, weight_decay=1e-6)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, transformer_state, obs):
        # print('TRANSFORM ACTOR!!!', transformer_state.shape)

        transformer_state = transformer_state.view(-1, self.input_dims * self.patch_length*self.seq_length)

        x = F.elu(self.ln1(self.fc1(transformer_state)))
        x = F.elu(self.ln2(self.fc2(x)))
        x = F.elu(self.ln3(self.fc3(x)))

        # alpha = F.elu(self.ln1_alpha(self.fc1_alpha(transformer_state.view(batch_size, -1))))
        # alpha = F.elu(self.ln2_alpha(self.fc2_alpha(alpha)))
        # alpha = F.elu(self.ln3_alpha(self.fc3_alpha(alpha)))

        # lever action
        pi_lever = F.softmax(self.pi_lever(x))

        # attention allocation
        # pi_sensor = self.pi_sensor(alpha)

        # Concatenate along the last dimension
        # pi = torch.cat((pi_lever, pi_sensor), dim=-1)

        return pi_lever

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.checkpoint_file))