import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# ---------------------------
# Custom LSTM Cell for Temporal Modeling
# ---------------------------
class CustomLSTMCell(nn.Module):
    def __init__(self, patch_size, d_model, dff):
        super(CustomLSTMCell, self).__init__()
        self.patch_size = patch_size
        self.d_model = d_model
        self.dff = dff
        self.input_size = self.d_model
        self.temperature = 1.0

        # Linear layers for input and hidden state transformations for each gate.
        self.WI = nn.Linear(self.dff, self.dff)
        self.WF = nn.Linear(self.dff, self.dff)
        self.WO = nn.Linear(self.dff, self.dff)
        self.WZ = nn.Linear(self.dff, self.dff)
        self.RI = nn.Linear(self.input_size, self.dff)
        self.RF = nn.Linear(self.input_size, self.dff)
        self.RO = nn.Linear(self.input_size, self.dff)
        self.RZ = nn.Linear(self.input_size, self.dff)

    def forward(self, Zi, Ci, Mi, Hi, Ni):
        # Reshape inputs.
        Zi = Zi.view(-1, self.patch_size, self.input_size)
        Ci = Ci.view(-1, self.patch_size, self.dff)
        Hi = Hi.view(-1, self.patch_size, self.dff)
        Ni = Ni.view(-1, self.patch_size, self.dff)

        # Compute gate pre-activations.
        I_tilde = self.WI(Hi) + self.RI(Zi)
        F_tilde = self.WF(Hi) + self.RF(Zi)
        O_tilde = self.WO(Hi) + self.RO(Zi)
        Z_tilde = self.WZ(Hi) + self.RZ(Zi)

        # Gate computations with noise for regularization.
        M_t = torch.max(F_tilde + Mi, I_tilde)
        diff = I_tilde - M_t
        I_t = torch.exp(diff / self.temperature)
        F_t = torch.exp(F_tilde + Mi - M_t)

        noise_scale = 0.01
        noisy_O_tilde = O_tilde + torch.randn_like(O_tilde) * noise_scale
        O_t = torch.sigmoid(noisy_O_tilde / self.temperature)

        noisy_Z_tilde = Z_tilde + torch.randn_like(Z_tilde) * noise_scale
        Z_t = torch.tanh(noisy_Z_tilde / self.temperature)

        # Update cell state and hidden state.
        C_t = Ci * F_t + I_t * Z_t
        H_t = O_t * (C_t / Ni)
        N_t = F_t * Ni + I_t

        return C_t, M_t, H_t, N_t

# ---------------------------
# Transformer Block: Multi-head Attention + Feed-Forward Network
# ---------------------------
class TransformerBlock(nn.Module):
    def __init__(self, beta, input_dims=64, hidden_dim=128, fc1_dims=64, fc2_dims=32, n_actions=4, nheads=6):
        super(TransformerBlock, self).__init__()
        self.input_dims = input_dims
        self.hidden_dim = hidden_dim
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.patch_length = 256
        self.n_actions = n_actions
        self.d_model = self.input_dims
        self.num_heads = nheads
        self.dff = 512
        self.dropout = 0.01
        self.d_k = self.d_model // self.num_heads

        # Linear layers for query, key, and value.
        self.W_q = nn.Linear(self.d_model, self.d_k * self.num_heads)
        self.W_k = nn.Linear(self.d_model, self.d_k * self.num_heads)
        self.W_v = nn.Linear(self.d_model, self.d_k * self.num_heads)

        # Additional linear layers combining LSTM hidden states.
        self.W_C1q = nn.Linear(self.dff * 3, self.d_k * self.num_heads)
        self.W_C1k = nn.Linear(self.dff * 3, self.d_k * self.num_heads)
        self.W_C1v = nn.Linear(self.dff * 3, self.d_k * self.num_heads)

        # Layer normalization layers.
        self.normQ = nn.LayerNorm(self.d_model)
        self.normK = nn.LayerNorm(self.d_model)
        self.normV = nn.LayerNorm(self.d_model)
        self.normQE = nn.LayerNorm(self.d_model)
        self.normKE = nn.LayerNorm(self.d_model)
        self.normVE = nn.LayerNorm(self.d_model)
        self.normQH = nn.LayerNorm(self.d_model)
        self.normKH = nn.LayerNorm(self.d_model)
        self.normVH = nn.LayerNorm(self.d_model)

        # Feed-forward network layers.
        self.linear1 = nn.Linear(self.num_heads * self.d_k, self.d_model * 5)
        self.linear2 = nn.Linear(self.d_model * 5, self.d_model)
        self.norm1 = nn.LayerNorm(self.d_model)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)

        # Residual connection scales.
        self.residual_scale_L1 = 1.0
        self.residual_scale_L2 = 1.0

        self.optimizer = optim.Adam(self.parameters(), lr=beta, betas=(0.9, 0.99), eps=1e-8, weight_decay=1e-6)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state, H1, H2, H3, layer):
        # Reshape input state and LSTM hidden states.
        state = state.view(-1, self.patch_length, self.input_dims)
        H1 = H1.view(-1, self.patch_length, self.dff)
        H2 = H2.view(-1, self.patch_length, self.dff)
        H3 = H3.view(-1, self.patch_length, self.dff)
        H = torch.cat((H1, H2, H3), dim=2)
        batch_size = state.shape[0]

        # Compute query, key, and value based on layer.
        if layer > 1:
            q = self.normQE(self.W_q(state)).view(batch_size, -1, self.num_heads, self.d_k) * \
                self.normQH(self.W_C1q(H)).view(batch_size, -1, self.num_heads, self.d_k)
            k = self.normKE(self.W_k(state)).view(batch_size, -1, self.num_heads, self.d_k) * \
                self.normKH(self.W_C1k(H)).view(batch_size, -1, self.num_heads, self.d_k)
            v = self.normVE(self.W_v(state)).view(batch_size, -1, self.num_heads, self.d_k) * \
                self.normVH(self.W_C1v(H)).view(batch_size, -1, self.num_heads, self.d_k)
        elif layer == 0:
            q = self.normQE(self.W_q(state)).view(batch_size, -1, self.num_heads, self.d_k) + \
                self.normQH(self.W_C1q(H)).view(batch_size, -1, self.num_heads, self.d_k * 4)
            k = self.normKE(self.W_k(state)).view(batch_size, -1, self.num_heads, self.d_k) + \
                self.normKH(self.W_C1k(H)).view(batch_size, -1, self.num_heads, self.d_k * 4)
            v = self.normVE(self.W_v(state)).view(batch_size, -1, self.num_heads, self.d_k) + \
                self.normVH(self.W_C1v(H)).view(batch_size, -1, self.num_heads, self.d_k * 4)
        elif layer == 1:
            q = self.normQE(self.W_q(state)).view(batch_size, -1, self.num_heads, self.d_k)
            k = self.normKE(self.W_k(state)).view(batch_size, -1, self.num_heads, self.d_k)
            v = self.normVE(self.W_v(state)).view(batch_size, -1, self.num_heads, self.d_k)

        # Transpose for multi-head attention.
        q, k, v = [x.transpose(1, 2) for x in (q, k, v)]  # (batch_size, num_heads, seq_len, d_k)
        attn_values, A = self.calculate_attention(q, k, v)
        attn_values = attn_values.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)

        # Apply residual connection and feed-forward network.
        Z1 = state + self.residual_scale_L1 * self.dropout1(attn_values)
        Z2 = Z1
        Z2_norm = self.norm1(Z2)
        Z3 = F.gelu(self.linear1(Z2_norm))
        Z4 = self.linear2(Z3)
        Z5 = Z2 + self.residual_scale_L2 * self.dropout2(Z4)
        return Z5

    def calculate_attention(self, q, k, v):
        # Compute scaled dot-product attention.
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        A = F.softmax(scores, dim=-1)
        return torch.matmul(A, v), A


# ---------------------------
# Time Embedding for Temporal Information
# ---------------------------
class TimeEmbedding(nn.Module):
    def __init__(self, embedding_dim):
        super(TimeEmbedding, self).__init__()
        self.embedding_dim = embedding_dim

    def forward(self, t):
        device = t.device
        half_dim = self.embedding_dim // 2
        emb = torch.exp(-torch.arange(half_dim, device=device).float() * (np.log(10000) / (half_dim - 1)))
        emb = t.float().unsqueeze(1) * emb.unsqueeze(0)  # (batch_size, half_dim)
        emb = torch.cat((torch.sin(emb), torch.cos(emb)), dim=-1)  # (batch_size, embedding_dim)
        return emb


# ---------------------------
# Transformer Network for the Video Autoencoder
# ---------------------------
class TransformerNetwork(nn.Module):
    def __init__(self, beta, input_dims=15*20*3, hidden_dim=128, fc1_dims=64, fc2_dims=32, n_actions=4):
        super(TransformerNetwork, self).__init__()
        # Set input dimensions (flattened patch size) and other parameters.
        self.input_dims = input_dims
        self.hidden_dim = hidden_dim
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.patch_length = 256  # Number of patches.
        self.n_actions = n_actions
        self.d_model = self.input_dims
        self.num_heads = 1
        self.dff = 512
        self.dropout = 0.01

        # LSTM cells for temporal processing.
        self.LSTM1 = CustomLSTMCell(patch_size=self.patch_length, d_model=self.input_dims, dff=self.dff)
        self.LSTM2 = CustomLSTMCell(patch_size=self.patch_length, d_model=self.input_dims, dff=self.dff)
        self.LSTM3 = CustomLSTMCell(patch_size=self.patch_length, d_model=self.input_dims, dff=self.dff)

        # Transformer blocks for encoding.
        self.transformer_block1 = TransformerBlock(beta, input_dims=self.input_dims, hidden_dim=self.hidden_dim,
                                                    fc1_dims=self.fc1_dims, fc2_dims=self.fc2_dims,
                                                    n_actions=self.n_actions, nheads=10)
        self.transformer_block2 = TransformerBlock(beta, input_dims=self.input_dims, hidden_dim=self.hidden_dim,
                                                    fc1_dims=self.fc1_dims, fc2_dims=self.fc2_dims,
                                                    n_actions=self.n_actions, nheads=10)
        self.transformer_block3 = TransformerBlock(beta, input_dims=self.input_dims, hidden_dim=self.hidden_dim,
                                                    fc1_dims=self.fc1_dims, fc2_dims=self.fc2_dims,
                                                    n_actions=self.n_actions, nheads=10)
        # Transformer block for latent mean.
        self.transformer_ZMU = TransformerBlock(beta, input_dims=self.input_dims, hidden_dim=self.hidden_dim,
                                                 fc1_dims=self.fc1_dims, fc2_dims=self.fc2_dims,
                                                 n_actions=self.n_actions, nheads=10)
        self.WZMU = nn.Linear(self.input_dims, self.input_dims)

        # Decoder components.
        self.WZExpansion = nn.Linear(self.input_dims, self.input_dims)
        self.transformer_ZP1 = TransformerBlock(beta, input_dims=self.input_dims, hidden_dim=self.hidden_dim,
                                                fc1_dims=self.fc1_dims, fc2_dims=self.fc2_dims,
                                                n_actions=self.n_actions, nheads=10)
        self.LSTMR1 = CustomLSTMCell(patch_size=self.patch_length, d_model=self.input_dims, dff=self.dff)
        self.transformer_ZP2 = TransformerBlock(beta, input_dims=self.input_dims, hidden_dim=self.hidden_dim,
                                                fc1_dims=self.fc1_dims, fc2_dims=self.fc2_dims,
                                                n_actions=self.n_actions, nheads=10)
        self.LSTMR2 = CustomLSTMCell(patch_size=self.patch_length, d_model=self.input_dims, dff=self.dff)
        self.transformer_ZP3 = TransformerBlock(beta, input_dims=self.input_dims, hidden_dim=self.hidden_dim,
                                                fc1_dims=self.fc1_dims, fc2_dims=self.fc2_dims,
                                                n_actions=self.n_actions, nheads=10)
        self.LSTMR3 = CustomLSTMCell(patch_size=self.patch_length, d_model=self.input_dims, dff=self.dff)

        # Transformer encoder layers used in the decoder.
        self.TE1 = nn.TransformerEncoderLayer(
            d_model=self.input_dims,
            nhead=10,
            dim_feedforward=512,
            dropout=0.01,
            activation='gelu'
        )
        self.TE2 = nn.TransformerEncoderLayer(
            d_model=self.input_dims,
            nhead=10,
            dim_feedforward=512,
            dropout=0.01,
            activation='gelu'
        )
        self.WZOUT = nn.Linear(self.input_dims, self.input_dims)

        self.optimizer = optim.Adam(self.parameters(), lr=beta, betas=(0.9, 0.99),
                                      eps=1e-8, weight_decay=1e-6)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def encoder(self, state, C1, M1, H1, N1,
                      C2, M2, H2, N2,
                      C3, M3, H3, N3):
        # Reshape state for the transformer: (batch_size, 1, patch_length, input_dims)
        state = state.view(-1, 1, self.patch_length, self.input_dims)
        # Pass through transformer blocks and update LSTM cells.
        Z = self.transformer_block1(state, H1, H2, H3, layer=2)
        C1, M1, H1, N1 = self.LSTM1(Z, C1, M1, H1, N1)
        Z = self.transformer_block2(Z, H1, H2, H3, layer=2)
        C2, M2, H2, N2 = self.LSTM2(Z, C2, M2, H2, N2)
        Z = self.transformer_block3(Z, H1, H2, H3, layer=2)
        C3, M3, H3, N3 = self.LSTM3(Z, C3, M3, H3, N3)
        # Get latent mean representation.
        Z = self.transformer_ZMU(Z, H1, H2, H3, layer=2)
        Z = self.WZMU(Z)
        return C1, M1, H1, N1, C2, M2, H2, N2, C3, M3, H3, N3, Z

    def decoder(self, Z, C1, M1, H1, N1,
                      C2, M2, H2, N2,
                      C3, M3, H3, N3, t):
        # Expand the latent representation.
        Z = Z.view(-1, self.patch_length, self.input_dims)
        Z = self.WZExpansion(Z).view(-1, self.patch_length, self.input_dims)
        batch_size = Z.shape[0]
        outBackwards = []
        # Decode in a loop until t reaches 0.
        while t >= 0:
            Z = self.transformer_ZP1(Z, H1, H2, H3, layer=2)
            C1, M1, H1, N1 = self.LSTMR1(Z, C1, M1, H1, N1)
            Z = self.transformer_ZP2(Z, H1, H2, H3, layer=2)
            C2, M2, H2, N2 = self.LSTMR2(Z, C2, M2, H2, N2)
            Z = self.transformer_ZP3(Z, H1, H2, H3, layer=2)
            C3, M3, H3, N3 = self.LSTMR3(Z, C3, M3, H3, N3)
            Z = self.TE1(Z)
            Z = self.TE2(Z)
            Z = torch.sigmoid(self.WZOUT(Z))
            outBackwards.append(Z.view(batch_size, 1, self.patch_length, self.input_dims))
            t -= 1
        # Reverse the time order and concatenate.
        out = torch.cat(outBackwards[::-1], dim=1)
        return out

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, C1, M1, H1, N1,
                      C2, M2, H2, N2,
                      C3, M3, H3, N3, t):
        # Encoder pass.
        C1, M1, H1, N1, C2, M2, H2, N2, C3, M3, H3, N3, Z = \
            self.encoder(x, C1, M1, H1, N1, C2, M2, H2, N2, C3, M3, H3, N3)
        # Decoder is applied only at the final time step.
        if t == 49:
            decodedZ = self.decoder(Z, C1, M1, H1, N1, C2, M2, H2, N2, C3, M3, H3, N3, t)
        else:
            decodedZ = 0
        return C1, M1, H1, N1, C2, M2, H2, N2, C3, M3, H3, N3, decodedZ

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(torch.load(self.checkpoint_file))
