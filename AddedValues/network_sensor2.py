import os
import torch
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Tuple


# --- Base Class for Models ---

class BaseModel(nn.Module):
    """A base model class with saving and loading capabilities."""

    def __init__(self, name: str, chkpt_dir: str):
        super().__init__()
        self.name = name
        self.checkpoint_dir = chkpt_dir
        # Append '_td3' to match the original file naming convention
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name + '_td3')
        # Ensure the checkpoint directory exists
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

    def save_checkpoint(self):
        """Saves the model state dictionary to a file."""
        print(f'... saving checkpoint for {self.name} ...')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        """Loads the model state dictionary from a file."""
        print(f'... loading checkpoint for {self.name} ...')
        if os.path.exists(self.checkpoint_file):
            self.load_state_dict(T.load(self.checkpoint_file))
        else:
            print(f'Warning: Checkpoint file not found for {self.name} at {self.checkpoint_file}')


# --- Custom Recurrent Cell ---

class CustomLSTMCell(nn.Module):
    """
    A custom LSTM-like cell with a unique gating mechanism.
    This cell maintains multiple state variables: C (cell), M (max), H (hidden), and N (normalization).
    """

    def __init__(self, patch_size: int, d_model: int, dff: int = 1024):
        super(CustomLSTMCell, self).__init__()
        self.patch_size = patch_size
        self.d_model = d_model
        self.dff = dff

        # Linear transformations for the gates from the hidden state (H)
        self.wi = nn.Linear(self.dff, self.dff)
        self.wf = nn.Linear(self.dff, self.dff)
        self.wo = nn.Linear(self.dff, self.dff)
        self.wz = nn.Linear(self.dff, self.dff)

        # Linear transformations for the gates from the input (Z)
        self.ri = nn.Linear(self.d_model, self.dff)
        self.rf = nn.Linear(self.d_model, self.dff)
        self.ro = nn.Linear(self.d_model, self.dff)
        self.rz = nn.Linear(self.d_model, self.dff)

    def forward(self, z_input: T.Tensor, c_prev: T.Tensor, m_prev: T.Tensor,
                h_prev: T.Tensor, n_prev: T.Tensor, mask: T.Tensor) -> Tuple[T.Tensor, ...]:
        """Performs one step of the custom LSTM computation."""
        # Reshape inputs for processing
        mask = mask.view(-1, self.patch_size, 1)
        z_input = z_input.view(-1, self.patch_size, self.d_model)
        c_prev = c_prev.view(-1, self.patch_size, self.dff)
        h_prev = h_prev.view(-1, self.patch_size, self.dff)
        n_prev = n_prev.view(-1, self.patch_size, self.dff)

        # Gate computations
        i_tilde = self.wi(h_prev) + self.ri(z_input)
        f_tilde = self.wf(h_prev) + self.rf(z_input)
        o_tilde = self.wo(h_prev) + self.ro(z_input)
        z_tilde = self.wz(h_prev) + self.rz(z_input)

        # Custom gating mechanism using max for stability
        m_t = torch.max(f_tilde + m_prev, i_tilde)
        i_t = torch.exp(i_tilde - m_t)
        f_t = torch.exp(f_tilde + m_prev - m_t)

        o_t = torch.sigmoid(o_tilde)
        n_t = f_t * n_prev + i_t
        z_t = torch.tanh(z_tilde)

        # Update states based on gates and apply mask
        c_t = (c_prev * f_t + z_t * i_t) * mask + (1 - mask) * c_prev
        # Normalize the cell state before applying the output gate
        h_t = o_t * (c_t / (n_t + 1e-8)) * mask + (1 - mask) * h_prev

        return c_t, m_t, h_t, n_t


# --- Transformer Components ---

class TransformerBlock(nn.Module):
    """A custom transformer block that iteratively processes a sequence using a CustomLSTMCell."""

    def __init__(self, beta: float, hidden_dim: int, fc1_dims: int, fc2_dims: int,
                 name: str = 'transformer_block', chkpt_dir: str = 'td3_MAT'):
        super(TransformerBlock, self).__init__()
        self.token_dim = 128 + 4 + 8  # Dimension of a single token
        self.patch_length = 4
        self.seq_length = 8
        self.d_model = self.token_dim
        self.num_heads = 1
        self.dff = 1024
        self.dropout_rate = 0.01
        self.d_k = self.d_model // self.num_heads

        # Attention weights for input state
        self.w_q = nn.Linear(self.d_model, self.d_k * self.num_heads)
        self.w_k = nn.Linear(self.d_model, self.d_k * self.num_heads)
        self.w_v = nn.Linear(self.d_model, self.d_k * self.num_heads)

        # Attention weights for recurrent hidden state
        self.w_cq = nn.Linear(self.dff, self.d_k * self.num_heads)
        self.w_ck = nn.Linear(self.dff, self.d_k * self.num_heads)
        self.w_cv = nn.Linear(self.dff, self.d_k * self.num_heads)

        # Dropout and layer normalization
        self.dropout1 = nn.Dropout(self.dropout_rate * 2)
        self.norm1 = nn.LayerNorm(self.d_model)

        # Custom LSTM cell for recurrence
        self.lstm = CustomLSTMCell(patch_size=self.patch_length, d_model=self.d_model, dff=self.dff)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def _calculate_attention(self, q: T.Tensor, k: T.Tensor, v: T.Tensor) -> Tuple[T.Tensor, T.Tensor]:
        """Computes scaled dot-product attention."""
        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))
        attention_weights = F.softmax(scores, dim=-1)
        return torch.matmul(attention_weights, v), attention_weights

    def forward(self, state: T.Tensor, mask: T.Tensor) -> Tuple[T.Tensor, T.Tensor]:
        """Processes the input sequence iteratively."""
        state = state.view(-1, self.seq_length, self.patch_length, self.token_dim)
        mask = mask.view(-1, self.seq_length, self.patch_length, 1)
        batch_size = state.shape[0]

        # Initialize LSTM states
        c = torch.zeros(batch_size, self.patch_length, self.dff, device=self.device)
        m = torch.zeros(batch_size, self.patch_length, self.dff, device=self.device)
        h = torch.zeros(batch_size, self.patch_length, self.dff, device=self.device)
        n = torch.zeros(batch_size, self.patch_length, self.dff, device=self.device)

        attention_maps = []

        for i in range(self.seq_length):
            state_i = state[:, i, :, :]

            # Form query, key, and value by combining input and recurrent state
            q = self.w_q(state_i).view(batch_size, -1, self.num_heads, self.d_k) + self.w_cq(h).view(batch_size, -1,
                                                                                                     self.num_heads,
                                                                                                     self.d_k)
            k = self.w_k(state_i).view(batch_size, -1, self.num_heads, self.d_k) + self.w_ck(h).view(batch_size, -1,
                                                                                                     self.num_heads,
                                                                                                     self.d_k)
            v = self.w_v(state_i).view(batch_size, -1, self.num_heads, self.d_k) + self.w_cv(h).view(batch_size, -1,
                                                                                                     self.num_heads,
                                                                                                     self.d_k)

            q, k, v = [x.transpose(1, 2) for x in (q, k, v)]

            # Compute attention
            attn_values, attn_map = self._calculate_attention(q, k, v)
            attn_values = attn_values.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)
            attention_maps.append(attn_map)

            # Add & Norm, followed by LSTM update
            z1 = state_i + self.dropout1(attn_values)
            z2 = self.norm1(z1)

            mask_i = mask[:, i, :, :]
            c, m, h, n = self.lstm(z2, c, m, h, n, mask_i)

        # Stack attention maps from each time step
        attention_tensor = torch.stack(attention_maps, dim=1)
        return h, attention_tensor


class TransformerNetwork(BaseModel):
    """A network that embeds input and processes it with a TransformerBlock."""

    def __init__(self, beta: float, input_dims: int, hidden_dim: int, fc1_dims: int, fc2_dims: int,
                 name: str = 'transformer', chkpt_dir: str = 'td3_MAT'):
        super(TransformerNetwork, self).__init__(name, chkpt_dir)
        self.token_dim = 128 + 4 + 8
        self.patch_length = 4
        self.seq_length = 8

        # Input embedding layers
        self.fc_state_projection = nn.Linear(self.token_dim, self.token_dim)
        self.ln_state_projection = nn.LayerNorm(self.token_dim)
        self.fc2_state_projection = nn.Linear(self.token_dim, self.token_dim)
        self.ln2_state_projection = nn.LayerNorm(self.token_dim)

        # Transformer block
        self.transformer_block1 = TransformerBlock(beta, hidden_dim, fc1_dims, fc2_dims)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state: T.Tensor, t: T.Tensor, m: T.Tensor) -> T.Tensor:
        """Processes the state through embeddings and the transformer block."""
        batch_size = state.shape[0]
        # Reshape flat input into tokens for embedding
        state_flat = state.view(-1, self.token_dim)

        # Generate embeddings with residual connections
        x = F.elu(state_flat + self.ln_state_projection(self.fc_state_projection(state_flat)))
        state_embedded = self.ln2_state_projection(state_flat + self.fc2_state_projection(x))

        # Add small noise for regularization
        noise = torch.randn_like(state_embedded) * 0.001
        state_embedded += noise

        # Reshape back to sequence format for transformer
        state_reshaped = state_embedded.view(batch_size, self.seq_length, self.patch_length, self.token_dim)

        # In this implementation, only the final hidden state is used, not the attention maps.
        final_hidden_state, _ = self.transformer_block1(state_reshaped, m)

        return final_hidden_state


# --- Actor-Critic Networks ---

class CriticNetwork(BaseModel):
    """Critic Network that outputs a distribution over Q-values."""

    def __init__(self, beta: float, input_dims: int, n_actions: int, hidden_dim: int = 512, fc1_dims: int = 256,
                 fc2_dims: int = 128, name: str = 'critic', chkpt_dir: str = 'td3_MAT', num_q_bins: int = 10):
        super(CriticNetwork, self).__init__(name, chkpt_dir)
        self.recurrent_state_dim = 1024  # Output dimension of the CustomLSTMCell's hidden state
        self.patch_length = 4
        self.n_actions = n_actions
        self.flat_input_dims = self.recurrent_state_dim * self.patch_length

        # Projection layer for the action
        self.fc_action_projection = nn.Linear(n_actions, self.flat_input_dims)

        # Fully connected layers
        self.fc1 = nn.Linear(self.flat_input_dims * 2, hidden_dim)  # *2 for state + projected action
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, fc1_dims)
        self.ln2 = nn.LayerNorm(fc1_dims)
        self.fc3 = nn.Linear(fc1_dims, fc2_dims)
        self.ln3 = nn.LayerNorm(fc2_dims)

        # Output layer for the Q-value distribution
        self.q_dist_output = nn.Linear(fc2_dims, num_q_bins)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, transformer_state: T.Tensor, action: T.Tensor, obs: T.Tensor) -> T.Tensor:
        """Forward pass for the critic network."""
        state = transformer_state.view(-1, self.flat_input_dims)
        action_proj = self.fc_action_projection(action.view(-1, self.n_actions))

        state_action = torch.cat((state, action_proj), dim=1)

        x = F.elu(self.ln1(self.fc1(state_action)))
        x = F.elu(self.ln2(self.fc2(x)))
        x = F.elu(self.ln3(self.fc3(x)))

        q_dist = F.softmax(self.q_dist_output(x), dim=1)
        return q_dist


class ActorNetwork(BaseModel):
    """Actor Network that outputs a policy (action probabilities)."""

    def __init__(self, alpha: float, input_dims: int, n_actions: int, hidden_dim: int = 512, fc1_dims: int = 256,
                 fc2_dims: int = 128, name: str = 'Actor', chkpt_dir: str = 'td3_MAT'):
        super(ActorNetwork, self).__init__(name, chkpt_dir)
        self.recurrent_state_dim = 1024
        self.patch_length = 4
        self.n_actions = n_actions
        self.flat_input_dims = self.recurrent_state_dim * self.patch_length

        # Fully connected layers
        self.fc1 = nn.Linear(self.flat_input_dims, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, fc1_dims)
        self.ln2 = nn.LayerNorm(fc1_dims)
        self.fc3 = nn.Linear(fc1_dims, fc2_dims)
        self.ln3 = nn.LayerNorm(fc2_dims)

        # Output layer for the policy
        self.pi_output = nn.Linear(fc2_dims, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, transformer_state: T.Tensor, obs: T.Tensor) -> T.Tensor:
        """Forward pass for the actor network."""
        state = transformer_state.view(-1, self.flat_input_dims)

        x = F.elu(self.ln1(self.fc1(state)))
        x = F.elu(self.ln2(self.fc2(x)))
        x = F.elu(self.ln3(self.fc3(x)))

        action_probs = F.softmax(self.pi_output(x), dim=1)
        return action_probs
