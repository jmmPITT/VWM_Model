#!/usr/bin/env python3

"""
-------------------------------------------------------------------------------
Cleaned-up code for Custom LSTM cells, Transformer blocks, and Actor/Critic 
networks. The functionality remains exactly the same, and variable names are 
unchanged. Only formatting, code organization, docstrings, and comments have 
been improved or added.
-------------------------------------------------------------------------------
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.utils.rnn as rnn_utils


class CustomLSTMCell(nn.Module):
    """
    A custom LSTM-like cell for working memory updates. Incorporates
    exponential transformations (exp) for input and forget gates, and
    includes additional gating for memory updates.

    :param patch_size: Number of patches/tokens in spatial dimension.
    :param d_model: Model dimension (features) per patch.
    """
    def __init__(self, patch_size, d_model):
        super(CustomLSTMCell, self).__init__()
        self.patch_size = patch_size
        self.d_model = d_model
        self.dff = 1024  # feedforward dimension
        self.input_size = self.d_model

        # Linear transformations for hidden state (H) in gating
        self.WI = nn.Linear(self.dff, self.dff)
        self.WF = nn.Linear(self.dff, self.dff)
        self.WO = nn.Linear(self.dff, self.dff)
        self.WZ = nn.Linear(self.dff, self.dff)

        # Linear transformations for input (Z) in gating
        self.RI = nn.Linear(self.input_size, self.dff)
        self.RF = nn.Linear(self.input_size, self.dff)
        self.RO = nn.Linear(self.input_size, self.dff)
        self.RZ = nn.Linear(self.input_size, self.dff)

    def forward(self, Zi, Ci, Mi, Hi, Ni, m):
        """
        Forward pass of the CustomLSTMCell.

        :param Zi: Current input embedding [batch, patch_size, d_model].
        :param Ci: Previous cell state C [batch, patch_size, dff].
        :param Mi: Previous memory state M [batch, patch_size, dff].
        :param Hi: Previous hidden state H [batch, patch_size, dff].
        :param Ni: Previous normalizing factor N [batch, patch_size, dff].
        :param m:   Mask (or gating) indicator [batch, patch_size, 1].
        :return: (C_t, M_t, H_t, N_t) Updated states.
        """
        # Reshape the inputs/states for operation
        m = m.view(-1, self.patch_size, 1)
        Zi = Zi.view(-1, self.patch_size, self.input_size)
        Ci = Ci.view(-1, self.patch_size, self.dff)
        Hi = Hi.view(-1, self.patch_size, self.dff)
        Ni = Ni.view(-1, self.patch_size, self.dff)

        C_prev = Ci
        M_prev = Mi
        H_prev = Hi
        N_prev = Ni

        # Compute gating components
        I_tilde = self.WI(H_prev) + self.RI(Zi)
        F_tilde = self.WF(H_prev) + self.RF(Zi)
        O_tilde = self.WO(H_prev) + self.RO(Zi)
        Z_tilde = self.WZ(H_prev) + self.RZ(Zi)

        # Memory gate transformation
        M_t = torch.max(F_tilde + M_prev, I_tilde)  # scaling
        I_t = torch.exp(I_tilde - M_t)
        F_t = torch.exp(F_tilde + M_prev - M_t)

        # Output gate
        O_t = torch.sigmoid(O_tilde)
        # Normalizing factor
        N_t = F_t * N_prev + I_t
        # Activation
        Z_t = torch.tanh(Z_tilde)

        # Cell update with masking
        C_t = (C_prev * F_t + Z_t * I_t) * m + (1 - m) * C_prev
        # Hidden state update with masking
        H_t = O_t * (C_t / N_t) * m + (1 - m) * H_prev

        return C_t, M_t, H_t, N_t


class TransformerBlock(nn.Module):
    """
    A custom Transformer block that uses a single-head self-attention
    and a CustomLSTMCell for maintaining hidden states. This block 
    processes an 8-step sequence of 4 patches each, with the capacity
    for re-scaling attention via learned linear mappings.

    :param beta: Learning rate for the optimizer.
    :param input_dims: Input dimension for each patch token.
    :param hidden_dim: Hidden dimension in intermediate layers.
    :param fc1_dims: Dimension of first fully connected layer.
    :param fc2_dims: Dimension of second fully connected layer.
    :param n_actions: Number of possible actions (if used with RL).
    :param name: Name for checkpointing.
    :param chkpt_dir: Directory to save/load checkpoints.
    """
    def __init__(self, beta, input_dims=32, hidden_dim=128, 
                 fc1_dims=64, fc2_dims=32, n_actions=4,
                 name='transformer', chkpt_dir='td3_MAT'):
        super(TransformerBlock, self).__init__()
        self.input_dims = (128 + 4 + 9)
        self.hidden_dim = hidden_dim
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.patch_length = 4
        self.seq_length = 9
        self.n_actions = n_actions
        self.d_model = self.input_dims
        self.num_heads = 1
        self.d_k = self.d_model // self.num_heads
        self.dff = 1024
        self.dropout = 0.01

        # Linear layers for query, key, value for the input
        self.W_q = nn.Linear(self.d_model, self.d_k * self.num_heads)
        self.W_k = nn.Linear(self.d_model, self.d_k * self.num_heads)
        self.W_v = nn.Linear(self.d_model, self.d_k * self.num_heads)

        # Linear layers for query, key, value for hidden memory (C/H states)
        self.W_Cq = nn.Linear(self.dff, self.d_k * self.num_heads)
        self.W_Ck = nn.Linear(self.dff, self.d_k * self.num_heads)
        self.W_Cv = nn.Linear(self.dff, self.d_k * self.num_heads)

        # Position wise feed-forward layers
        self.linear1 = nn.Linear(self.num_heads * self.d_k, self.dff)
        self.linear2 = nn.Linear(self.dff, self.d_model)
        self.dropout1 = nn.Dropout(self.dropout * 2)

        # Normalization layers
        self.norm1 = nn.LayerNorm(self.d_model)
        self.norm2 = nn.LayerNorm(self.d_model)
        self.dropout2 = nn.Dropout(self.dropout)

        # Custom LSTM cell for memory updates
        self.LSTM = CustomLSTMCell(patch_size=self.patch_length, d_model=self.d_model)

        # Optimizer
        self.optimizer = optim.Adam(
            self.parameters(),
            lr=beta,
            betas=(0.9, 0.99),
            eps=1e-8,
            weight_decay=1e-6
        )
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state, m):
        """
        Forward pass through the TransformerBlock. Applies single-head 
        attention plus a custom LSTM memory update across seq_length steps.

        :param state: [batch, seq_length, patch_length, input_dims].
        :param m:     [batch, seq_length, patch_length, 1], mask/gate.
        :return: 
            - H: The updated hidden state after processing the entire sequence.
            - A_tensor: Attention weights over the sequence.
        """
        # Reshape state and mask
        state = state.view(-1, self.seq_length, self.patch_length, self.input_dims)
        m = m.view(-1, self.seq_length, self.patch_length, 1)

        batch_size = state.shape[0]

        # Initialize LSTM states
        C = torch.zeros(batch_size, self.patch_length, self.dff).to(self.device)
        M = torch.zeros(batch_size, self.patch_length, self.dff).to(self.device)
        H = torch.zeros(batch_size, self.patch_length, self.dff).to(self.device)
        N = torch.zeros(batch_size, self.patch_length, self.dff).to(self.device)

        A_list = []  # Keep track of attention weights at each step

        for i in range(self.seq_length):
            state_i = state[:, i, :, :].view(batch_size, self.patch_length, self.input_dims)

            # Construct query, key, value from input + hidden states
            q = self.W_q(state_i).view(batch_size, -1, self.num_heads, self.d_k) \
                * self.W_Cq(H).view(batch_size, -1, self.num_heads, self.d_k)
            k = self.W_k(state_i).view(batch_size, -1, self.num_heads, self.d_k) \
                * self.W_Ck(H).view(batch_size, -1, self.num_heads, self.d_k)
            v = self.W_v(state_i).view(batch_size, -1, self.num_heads, self.d_k) \
                * self.W_Cv(H).view(batch_size, -1, self.num_heads, self.d_k)

            # Rearrange dimensions for attention
            q, k, v = [x.transpose(1, 2) for x in (q, k, v)]

            # Compute attention
            attn_values, A = self.calculate_attention(q, k, v, m[:, i, :, :])
            attn_values = attn_values.transpose(1, 2).contiguous() \
                .view(batch_size, -1, self.num_heads * self.d_k)

            # Residual + dropout + layer norm
            Z1 = state_i + self.dropout1(attn_values)
            Z2 = self.norm1(Z1)

            # Mask for LSTM
            m_i = m[:, i, :, :].view(-1, self.patch_length, 1)

            # Update LSTM states with masked gating
            C, M, H, N = self.LSTM(Z2, C, M, H, N, m_i)

            A_list.append(A)

        # Stack attention maps across time steps
        A_tensor = torch.stack(A_list, dim=1)

        return H, A_tensor

    def calculate_attention(self, q, k, v, mask):
        """
        Computes scaled dot-product attention. 
        For single-head attention with optional masking.

        :param q: Query tensor [batch, num_heads, patch_length, d_k].
        :param k: Key tensor   [batch, num_heads, patch_length, d_k].
        :param v: Value tensor [batch, num_heads, patch_length, d_k].
        :param mask: Unused or partial mask [batch, patch_length, 1].
        :return: (attn_output, attention_weights)
        """
        d_k_sqrt = torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))
        scores = torch.matmul(q, k.transpose(-2, -1)) / d_k_sqrt
        A = F.softmax(scores, dim=-1)
        return torch.matmul(A, v), A


class TransformerNetwork(nn.Module):
    """
    A higher-level Transformer network that leverages the TransformerBlock 
    for multi-step processing, plus some linear projections for the input 
    embeddings. Gaussian noise is also added to the state embeddings.

    :param beta: Learning rate for the optimizer.
    :param input_dims: Input dimension for each token.
    :param hidden_dim: Hidden dimension for feedforward layers.
    :param fc1_dims: First linear layer dimension.
    :param fc2_dims: Second linear layer dimension.
    :param n_actions: Number of possible actions (RL setting).
    :param name: Name for checkpointing.
    :param chkpt_dir: Directory path for saving/loading checkpoints.
    """
    def __init__(self, beta, input_dims=32, hidden_dim=128, fc1_dims=64, 
                 fc2_dims=32, n_actions=4, name='transformer', chkpt_dir='td3_MAT'):
        super(TransformerNetwork, self).__init__()
        self.input_dims = (128 + 4 + 9)
        self.hidden_dim = hidden_dim
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.patch_length = 4
        self.seq_length = 9
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = name + '_td3'
        self.d_model = self.input_dims
        self.num_heads = 1
        self.dff = 1024
        self.dropout = 0.01

        # Linear + LayerNorm for initial state projection
        self.fc_state_projection = nn.Linear(self.input_dims, self.input_dims)
        self.ln_state_projection = nn.LayerNorm(self.input_dims)

        self.fc2_state_projection = nn.Linear(self.input_dims, self.input_dims)
        self.ln2_state_projection = nn.LayerNorm(self.input_dims)

        # Core TransformerBlock
        self.transformer_block1 = TransformerBlock(
            beta, input_dims=128, hidden_dim=self.hidden_dim,
            fc1_dims=self.fc1_dims, fc2_dims=self.fc2_dims,
            n_actions=self.n_actions
        )

        # Optimizer
        self.optimizer = optim.Adam(
            self.parameters(),
            lr=beta,
            betas=(0.9, 0.99),
            eps=1e-8,
            weight_decay=1e-6
        )
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state, m):
        """
        Forward pass of the TransformerNetwork. Projects input state 
        embeddings, adds noise, and then feeds them into a TransformerBlock.

        :param state: [batch, seq_length, patch_length, input_dims].
        :param m:     [batch, seq_length, patch_length, 1], mask/gate.
        :return: 
            - src: The final hidden states after TransformerBlock. 
            - A1:  The attention weights computed inside TransformerBlock.
        """
        # Flatten for projection
        state = state.view(-1, self.seq_length, self.patch_length, self.input_dims)
        batch_size = state.shape[0]
        state = state.view(batch_size * self.seq_length * self.patch_length, self.input_dims)

        # Nonlinear + residual projection
        X = F.elu(state + self.ln_state_projection(self.fc_state_projection(state)))
        state = self.ln2_state_projection(state + self.fc2_state_projection(X))

        # Add Gaussian noise to embeddings
        std = 0.001  # Standard deviation of the noise
        noise = torch.randn_like(state) * std
        state = state + noise

        # Reshape back to [batch, seq_length, patch_length, input_dims]
        state = state.view(batch_size, self.seq_length, self.patch_length, self.input_dims)

        src, A1 = self.transformer_block1(state, m)
        return src, A1

    def save_checkpoint(self):
        """
        Save model parameters to checkpoint file.
        """
        print('... saving checkpoint ...')
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        """
        Load model parameters from checkpoint file.
        """
        print('... loading checkpoint ...')
        self.load_state_dict(torch.load(self.checkpoint_file))


class CriticNetwork(nn.Module):
    """
    A Critic network used in actor-critic RL setups (e.g., TD3). Takes in the 
    transformer's hidden state and an action, then outputs Q-values.

    :param beta: Learning rate.
    :param input_dims: Flattened dimension of the transformer's output per patch.
    :param hidden_dim: Dimension of hidden layers.
    :param fc1_dims: Size of the first fully connected layer.
    :param fc2_dims: Size of the second fully connected layer.
    :param n_actions: Number of actions. 
    :param name: Name for checkpointing.
    :param chkpt_dir: Directory path for checkpoints.
    """
    def __init__(self, beta, input_dims=32, hidden_dim=512, fc1_dims=256, fc2_dims=128,
                 n_actions=4, name='critic', chkpt_dir='td3_MAT'):
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

        # Project action vector into same dimensionality as the transformer's output
        self.fc_action_lever_projection = nn.Linear(
            self.n_actions, self.input_dims * self.patch_length * self.seq_length
        )
        self.ln_action_lever_projection = nn.LayerNorm(
            self.input_dims * self.patch_length * self.seq_length
        )

        # Fully connected layers for state-action representation
        self.fc1 = nn.Linear(
            self.input_dims * self.patch_length * self.seq_length +
            self.input_dims * self.patch_length * self.seq_length,
            hidden_dim
        )
        self.ln1 = nn.LayerNorm(hidden_dim)

        self.fc2 = nn.Linear(hidden_dim, fc1_dims)
        self.ln2 = nn.LayerNorm(fc1_dims)

        self.fc3 = nn.Linear(fc1_dims, fc2_dims)
        self.ln3 = nn.LayerNorm(fc2_dims)

        # Final layer outputs Q-value distribution
        self.q = nn.Linear(fc2_dims, 10)

        # Optimizer
        self.optimizer = optim.Adam(
            self.parameters(), 
            lr=beta,
            betas=(0.9, 0.99),
            eps=1e-8,
            weight_decay=1e-6
        )
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, transformer_state, action):
        """
        Forward pass of the Critic network.

        :param transformer_state: Output from the Transformer network. 
                                  [batch, input_dims * patch_length * seq_length].
        :param action: Action vector [batch, n_actions].
        :return: Softmax Q-values [batch, 10].
        """
        transformer_state = transformer_state.view(-1, self.input_dims * self.patch_length * self.seq_length)
        action = action.view(-1, self.n_actions)

        # Project and concatenate
        action_lever = self.fc_action_lever_projection(action)
        state_action = torch.cat((transformer_state, action_lever), dim=1)

        # Feedforward layers
        x = F.elu(self.ln1(self.fc1(state_action)))
        x = F.elu(self.ln2(self.fc2(x)))
        x = F.elu(self.ln3(self.fc3(x)))
        q = self.q(x)

        # Softmax output for distribution form (typical for some discrete RL)
        q = F.softmax(q, dim=1)
        return q

    def save_checkpoint(self):
        """
        Save critic network parameters.
        """
        print('... saving checkpoint ...')
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        """
        Load critic network parameters from file.
        """
        print('... loading checkpoint ...')
        self.load_state_dict(torch.load(self.checkpoint_file))


class ActorNetwork(nn.Module):
    """
    An Actor network for deciding actions in an actor-critic RL approach. 
    Consumes transformer's output and outputs a probability distribution 
    over possible actions.

    :param alpha: Learning rate for the optimizer.
    :param input_dims: Not used here (hard-coded to 1024).
    :param hidden_dim: Hidden dimension size.
    :param fc1_dims: Size of the first fully connected layer.
    :param fc2_dims: Size of the second fully connected layer.
    :param n_actions: Number of possible actions.
    :param name: Checkpoint name.
    :param chkpt_dir: Directory path for saving/loading.
    """
    def __init__(self, alpha, input_dims=16, hidden_dim=512, fc1_dims=256, fc2_dims=128,
                 n_actions=4, name='Actor', chkpt_dir='td3_MAT'):
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

        # Fully connected layers
        self.fc1 = nn.Linear(self.input_dims * self.patch_length * self.seq_length, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)

        self.fc2 = nn.Linear(hidden_dim, fc1_dims)
        self.ln2 = nn.LayerNorm(fc1_dims)

        self.fc3 = nn.Linear(fc1_dims, fc2_dims)
        self.ln3 = nn.LayerNorm(fc2_dims)

        # Final layer for policy distribution
        self.pi_lever = nn.Linear(fc2_dims, self.n_actions)

        # Optimizer
        self.optimizer = optim.Adam(
            self.parameters(),
            lr=alpha,
            betas=(0.9, 0.99),
            eps=1e-8,
            weight_decay=1e-6
        )
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, transformer_state):
        """
        Forward pass of the Actor network.

        :param transformer_state: [batch, input_dims * patch_length * seq_length].
        :return: Probability distribution over self.n_actions.
        """
        transformer_state = transformer_state.view(-1, self.input_dims * self.patch_length * self.seq_length)

        # Pass through fully connected layers
        x = F.elu(self.ln1(self.fc1(transformer_state)))
        x = F.elu(self.ln2(self.fc2(x)))
        x = F.elu(self.ln3(self.fc3(x)))

        # Output action probabilities
        pi_lever = F.softmax(self.pi_lever(x), dim=1)
        return pi_lever

    def save_checkpoint(self):
        """
        Save actor network parameters.
        """
        print('... saving checkpoint ...')
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        """
        Load actor network parameters from file.
        """
        print('... loading checkpoint ...')
        self.load_state_dict(torch.load(self.checkpoint_file))
