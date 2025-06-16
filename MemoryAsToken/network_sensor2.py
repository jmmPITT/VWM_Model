import os
import torch
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Tuple


# --- Base Class for Models ---

class BaseModel(nn.Module):
    """A base model class with saving and loading capabilities."""

    def __init__(self, name: str, chkpt_dir: str):
        super().__init__()
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

    def save_checkpoint(self):
        print(f'... saving checkpoint for {self.name} ...')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print(f'... loading checkpoint for {self.name} ...')
        if os.path.exists(self.checkpoint_file):
            self.load_state_dict(T.load(self.checkpoint_file))
        else:
            print(f'Warning: Checkpoint file not found for {self.name} at {self.checkpoint_file}')


# --- Custom Recurrent Cell ---

class CustomLSTMCell(nn.Module):
    def __init__(self, patch_size: int, d_model: int, dff: int = 1024):
        super().__init__()
        self.patch_size = patch_size
        self.d_model = d_model
        self.dff = dff
        self.wi = nn.Linear(dff, dff)
        self.wf = nn.Linear(dff, dff)
        self.wo = nn.Linear(dff, dff)
        self.wz = nn.Linear(dff, dff)
        self.ri = nn.Linear(d_model, dff)
        self.rf = nn.Linear(d_model, dff)
        self.ro = nn.Linear(d_model, dff)
        self.rz = nn.Linear(d_model, dff)

    def forward(self, z_input: T.Tensor, c_prev: T.Tensor, m_prev: T.Tensor,
                h_prev: T.Tensor, n_prev: T.Tensor, mask: T.Tensor) -> Tuple[T.Tensor, ...]:
        mask = mask.view(-1, self.patch_size, 1)
        z_input = z_input.view(-1, self.patch_size, self.d_model)
        c_prev = c_prev.view(-1, self.patch_size, self.dff)
        h_prev = h_prev.view(-1, self.patch_size, self.dff)
        n_prev = n_prev.view(-1, self.patch_size, self.dff)
        i_tilde = self.wi(h_prev) + self.ri(z_input)
        f_tilde = self.wf(h_prev) + self.rf(z_input)
        o_tilde = self.wo(h_prev) + self.ro(z_input)
        z_tilde = self.wz(h_prev) + self.rz(z_input)
        m_t = torch.max(f_tilde + m_prev, i_tilde)
        i_t = torch.exp(i_tilde - m_t)
        f_t = torch.exp(f_tilde + m_prev - m_t)
        o_t = F.sigmoid(o_tilde)
        n_t = f_t * n_prev + i_t
        z_t = F.tanh(z_tilde)
        c_t = (c_prev * f_t + z_t * i_t) * mask + (1 - mask) * c_prev
        h_t = o_t * (c_t / n_t) * mask + (1 - mask) * h_prev
        return c_t, m_t, h_t, n_t


# --- Transformer Components ---

class TransformerBlock(BaseModel):
    def __init__(self, name: str = 'transformer', chkpt_dir: str = 'td3_MAT', num_heads: int = 1, dff: int = 1024,
                 dropout: float = 0.01):
        super().__init__(name, chkpt_dir)
        # *** FIX: Hardcoded token_dim to the correct value (140) to avoid ambiguity. ***
        self.token_dim = 128 + 4 + 8  # This is the dimension of a single patch/token
        self.patch_length = 4
        self.seq_length = 8
        self.num_heads = num_heads
        self.d_k = self.token_dim // self.num_heads
        self.dff = dff

        self.w_q = nn.Linear(self.token_dim, self.d_k * self.num_heads)
        self.w_k = nn.Linear(self.token_dim, self.d_k * self.num_heads)
        self.w_v = nn.Linear(self.token_dim, self.d_k * self.num_heads)
        self.w_h_compress = nn.Linear(self.dff, self.token_dim)

        self.linear1 = nn.Linear(self.num_heads * self.d_k, self.dff)
        self.linear2 = nn.Linear(self.dff, self.token_dim)

        self.norm1 = nn.LayerNorm(self.token_dim)
        self.norm2 = nn.LayerNorm(self.token_dim)
        self.dropout1 = nn.Dropout(dropout * 2)
        self.dropout2 = nn.Dropout(dropout)

        self.lstm = CustomLSTMCell(patch_size=self.patch_length, d_model=self.token_dim, dff=self.dff)

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def _calculate_attention(self, q: T.Tensor, k: T.Tensor, v: T.Tensor) -> T.Tensor:
        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))
        return torch.matmul(F.softmax(scores, dim=-1), v)

    def forward(self, state: T.Tensor, mask: T.Tensor) -> T.Tensor:
        # *** FIX: Use self.token_dim for reshaping, not the ambiguous input_dims ***
        state = state.view(-1, self.seq_length, self.patch_length, self.token_dim)
        mask = mask.view(-1, self.seq_length, self.patch_length, 1)
        batch_size = state.shape[0]

        c = torch.zeros(batch_size, self.patch_length, self.dff, device=self.device)
        m = torch.zeros(batch_size, self.patch_length, self.dff, device=self.device)
        h = torch.zeros(batch_size, self.patch_length, self.dff, device=self.device)
        n = torch.zeros(batch_size, self.patch_length, self.dff, device=self.device)

        for i in range(self.seq_length):
            state_i = state[:, i, :, :]
            h_compressed = self.w_h_compress(h)
            # In this architecture, the query/key/value are based on a concatenation of the input
            # and the recurrent state. The length becomes patch_length * 2.
            concatenated_state = torch.cat((state_i, h_compressed), dim=1)

            q = self.w_q(concatenated_state).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
            k = self.w_k(concatenated_state).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
            v = self.w_v(concatenated_state).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

            attn_values = self._calculate_attention(q, k, v)
            attn_values = attn_values.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)

            z1 = state_i + self.dropout1(attn_values[:, :self.patch_length, :])
            z2 = self.norm1(z1)
            z3 = F.relu(self.linear1(z2))
            z4 = self.linear2(z3)
            z5 = self.norm2(z2 + self.dropout2(z4))

            mask_i = mask[:, i, :, :]
            c, m, h, n = self.lstm(z5, c, m, h, n, mask_i)

        return h


class TransformerNetwork(BaseModel):
    def __init__(self, beta: float, input_dims: int, hidden_dim: int, fc1_dims: int,
                 fc2_dims: int, name: str = 'transformer', chkpt_dir: str = 'td3_MAT'):
        super().__init__(name, chkpt_dir)
        # *** FIX: Define token_dim explicitly. The 'input_dims' (560) is for the entire observation. ***
        self.token_dim = 128 + 4 + 8
        self.patch_length = 4
        self.seq_length = 8

        self.fc_state_projection = nn.Linear(self.token_dim, self.token_dim)
        self.ln_state_projection = nn.LayerNorm(self.token_dim)
        self.fc2_state_projection = nn.Linear(self.token_dim, self.token_dim)
        self.ln2_state_projection = nn.LayerNorm(self.token_dim)

        self.transformer_block1 = TransformerBlock(name='t_block_1', chkpt_dir=chkpt_dir)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state: T.Tensor, t: T.Tensor, mask: T.Tensor) -> T.Tensor:
        batch_size = state.shape[0]
        # *** FIX: Reshape the incoming flattened observation (e.g., size 4480) into tokens of size 140 ***
        state_flat = state.view(-1, self.token_dim)

        x = F.elu(state_flat + self.ln_state_projection(self.fc_state_projection(state_flat)))
        state_embedded = self.ln2_state_projection(state_flat + self.fc2_state_projection(x))

        noise = torch.randn_like(state_embedded) * 0.001
        state_embedded += noise

        state_reshaped = state_embedded.view(batch_size, self.seq_length, self.patch_length, self.token_dim)

        processed_state = self.transformer_block1(state_reshaped, mask)

        return processed_state


# --- Actor-Critic Networks ---

class CriticNetwork(BaseModel):
    def __init__(self, beta: float, input_dims: int, hidden_dim: int, fc1_dims: int,
                 fc2_dims: int, n_actions: int, name: str = 'critic', chkpt_dir: str = 'td3_MAT',
                 num_q_bins: int = 10):
        super().__init__(name, chkpt_dir)
        # *** FIX: The input_dims for the critic is the output of the transformer, which is dff (1024) ***
        self.recurrent_state_dim = 1024
        self.patch_length = 4
        self.n_actions = n_actions
        self.flat_input_dims = self.recurrent_state_dim * self.patch_length

        self.fc_action_projection = nn.Linear(n_actions, self.flat_input_dims)

        self.fc1 = nn.Linear(self.flat_input_dims * 2, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, fc1_dims)
        self.ln2 = nn.LayerNorm(fc1_dims)
        self.fc3 = nn.Linear(fc1_dims, fc2_dims)
        self.ln3 = nn.LayerNorm(fc2_dims)
        self.q = nn.Linear(fc2_dims, num_q_bins)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, transformer_state: T.Tensor, action: T.Tensor, obs: T.Tensor) -> T.Tensor:
        state = transformer_state.view(-1, self.flat_input_dims)
        action_proj = self.fc_action_projection(action.view(-1, self.n_actions))

        state_action = torch.cat((state, action_proj), dim=1)

        x = F.elu(self.ln1(self.fc1(state_action)))
        x = F.elu(self.ln2(self.fc2(x)))
        x = F.elu(self.ln3(self.fc3(x)))

        return F.softmax(self.q(x), dim=1)


class ActorNetwork(BaseModel):
    def __init__(self, alpha: float, input_dims: int, hidden_dim: int, fc1_dims: int,
                 fc2_dims: int, n_actions: int, name: str = 'Actor', chkpt_dir: str = 'td3_MAT'):
        super().__init__(name, chkpt_dir)
        # *** FIX: The input_dims for the actor is the output of the transformer, which is dff (1024) ***
        self.recurrent_state_dim = 1024
        self.patch_length = 4
        self.n_actions = n_actions
        self.flat_input_dims = self.recurrent_state_dim * self.patch_length

        self.fc1 = nn.Linear(self.flat_input_dims, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, fc1_dims)
        self.ln2 = nn.LayerNorm(fc1_dims)
        self.fc3 = nn.Linear(fc1_dims, fc2_dims)
        self.ln3 = nn.LayerNorm(fc2_dims)
        self.pi = nn.Linear(fc2_dims, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, transformer_state: T.Tensor, obs: T.Tensor) -> T.Tensor:
        state = transformer_state.view(-1, self.flat_input_dims)

        x = F.elu(self.ln1(self.fc1(state)))
        x = F.elu(self.ln2(self.fc2(x)))
        x = F.elu(self.ln3(self.fc3(x)))

        return F.softmax(self.pi(x), dim=1)