import os
import torch
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torch.nn.utils.rnn as rnn_utils


class CustomLSTMCell(nn.Module):
    """Custom LSTM cell implementation with modifications for visual working memory.
    
    This cell implements a variant of LSTM with adaptations for handling visual input
    and maintaining working memory representations.
    """
    def __init__(self, patch_size, d_model):
        super(CustomLSTMCell, self).__init__()
        self.patch_size = patch_size
        self.d_model = d_model
        self.dff = 1024
        self.input_size = self.d_model

        # Linear transformations for gates
        # W* for hidden state transformations
        # R* for input transformations
        self.WI = nn.Linear(self.dff, self.dff)  # Input gate (hidden)
        self.WF = nn.Linear(self.dff, self.dff)  # Forget gate (hidden)
        self.WO = nn.Linear(self.dff, self.dff)  # Output gate (hidden)
        self.WZ = nn.Linear(self.dff, self.dff)  # Cell update (hidden)

        self.RI = nn.Linear(self.input_size, self.dff)  # Input gate (input)
        self.RF = nn.Linear(self.input_size, self.dff)  # Forget gate (input)
        self.RO = nn.Linear(self.input_size, self.dff)  # Output gate (input)
        self.RZ = nn.Linear(self.input_size, self.dff)  # Cell update (input)


    def forward(self, Zi, Ci, Mi, Hi, Ni, m):
        """Forward pass for the custom LSTM cell.
        
        Args:
            Zi: Input tensor
            Ci: Previous cell state
            Mi: Previous max gate value
            Hi: Previous hidden state
            Ni: Previous normalization factor
            m: Mask tensor for conditional updates
            
        Returns:
            tuple: Updated cell state, max gate value, hidden state, and normalization factor
        """
        # Reshape inputs to expected dimensions
        m = m.view(-1, self.patch_size, 1)
        Zi = Zi.view(-1, self.patch_size, self.input_size)
        Ci = Ci.view(-1, self.patch_size, self.dff)
        Hi = Hi.view(-1, self.patch_size, self.dff)
        Ni = Ni.view(-1, self.patch_size, self.dff)

        # Store previous states
        C_prev = Ci
        M_prev = Mi
        H_prev = Hi
        N_prev = Ni

        # Compute gate values
        I_tilde = self.WI(H_prev) + self.RI(Zi)  # Input gate pre-activation
        F_tilde = self.WF(H_prev) + self.RF(Zi)  # Forget gate pre-activation
        O_tilde = self.WO(H_prev) + self.RO(Zi)  # Output gate pre-activation
        Z_tilde = self.WZ(H_prev) + self.RZ(Zi)  # Cell update pre-activation

        # Log-sum-exp trick for numerical stability
        M_t = torch.max(F_tilde + M_prev, I_tilde)
        I_t = torch.exp(I_tilde - M_t)
        F_t = torch.exp(F_tilde + M_prev - M_t)

        # Compute gate activations and cell updates
        O_t = F.sigmoid(O_tilde)
        N_t = F_t*N_prev + I_t
        Z_t = F.tanh(Z_tilde)
        
        # Update cell state with mask
        C_t = (C_prev * F_t + Z_t * I_t) * m + (1-m) * C_prev
        
        # Compute hidden state
        H_t = O_t * (C_t/N_t)




        return C_t, M_t, H_t, N_t

class TransformerBlock(nn.Module):
    """Transformer encoder block with self-attention and LSTM memory mechanism.
    
    This block implements a modified transformer architecture with:
    1. Multi-head self-attention with context-dependent modulation
    2. Integration with a custom LSTM cell for maintaining working memory
    3. Skip connections and layer normalization
    """
    def __init__(self, beta, input_dims=32, hidden_dim=128, fc1_dims=64, fc2_dims=32, n_actions=4,
                 name='transformer', chkpt_dir='td3_MAT'):
        super(TransformerBlock, self).__init__()
        # Input and output dimensions
        self.input_dims = (128 + 4 + 8)  # VAE embedding + position encoding + temporal encoding
        self.hidden_dim = hidden_dim
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.patch_length = 4      # Number of visual patches (spatial tokens)
        self.seq_length = 8        # Maximum sequence length for temporal processing
        self.n_actions = n_actions
        
        # Self-attention parameters
        self.d_model = self.input_dims 
        self.num_heads = 1         # Single attention head
        self.dff = 1024            # Dimension of feed-forward network
        self.dropout = 0.01        # Dropout rate
        self.d_k = self.d_model // self.num_heads  # Dimension per head

        # Query, key, and value projection matrices
        self.W_q = nn.Linear(self.d_model, self.d_k*self.num_heads)
        self.W_k = nn.Linear(self.d_model, self.d_k*self.num_heads)
        self.W_v = nn.Linear(self.d_model, self.d_k*self.num_heads)

        # Context-dependent attention modulation
        self.W_Cq = nn.Linear(self.dff, self.d_k * self.num_heads)  # Context for query
        self.W_Ck = nn.Linear(self.dff, self.d_k * self.num_heads)  # Context for key
        self.W_Cv = nn.Linear(self.dff, self.d_k * self.num_heads)  # Context for value

        # Feed-forward network after attention
        self.linear1 = nn.Linear(self.num_heads * self.d_k, self.dff)
        self.linear2 = nn.Linear(self.dff, self.d_model)
        self.dropout1 = nn.Dropout(self.dropout*2)

        # Layer normalization and dropout
        self.norm1 = nn.LayerNorm(self.d_model)  # Normalization before LSTM
        self.norm2 = nn.LayerNorm(self.d_model)  # Normalization after feed-forward
        self.dropout2 = nn.Dropout(self.dropout)


        self.LSTM = CustomLSTMCell(patch_size=self.patch_length, d_model=self.d_model)




        self.optimizer = optim.Adam(self.parameters(), lr=beta, betas=(0.9, 0.99), eps=1e-8, weight_decay=1e-6)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state, m, kin):
        """Forward pass through the transformer block.
        
        Args:
            state: Input state tensor
            m: Mask tensor for conditional processing
            kin: Additional modulation parameter for attention
            
        Returns:
            tuple: (hidden_state, attention_weights)
        """
        # Reshape input state and mask
        state = state.view(-1, self.seq_length, self.patch_length, self.input_dims)
        m = m.view(-1, self.seq_length, self.patch_length, 1)
        batch_size = state.shape[0]
        
        # Initialize LSTM cell states
        C = torch.zeros(batch_size, self.patch_length, self.dff).to(self.device)  # Cell state
        M = torch.zeros(batch_size, self.patch_length, self.dff).to(self.device)  # Max gate values
        H = torch.zeros(batch_size, self.patch_length, self.dff).to(self.device)  # Hidden state
        N = torch.zeros(batch_size, self.patch_length, self.dff).to(self.device)  # Normalization factor
        
        # Initialize list to collect attention matrices
        A_list = []
        
        # Process each timestep in the sequence
        for i in range(self.seq_length):
            # Extract state at current timestep
            state_i = state[:, i, :, :].view(batch_size, self.patch_length, self.input_dims)
            
            # Compute context-modulated attention components
            # Multiply query/key/value projections by context-dependent factors from hidden state
            q = self.W_q(state_i).view(batch_size, -1, self.num_heads, self.d_k) * self.W_Cq(H).view(batch_size, -1, self.num_heads, self.d_k)
            k = self.W_k(state_i).view(batch_size, -1, self.num_heads, self.d_k) * self.W_Ck(H).view(batch_size, -1, self.num_heads, self.d_k)
            v = self.W_v(state_i).view(batch_size, -1, self.num_heads, self.d_k) * self.W_Cv(H).view(batch_size, -1, self.num_heads, self.d_k)
            
            # Reshape for attention calculation
            q, k, v = [x.transpose(1, 2) for x in (q, k, v)]  # [batch_size, num_heads, seq_len, d_k]
            
            # Compute attention
            attn_values, A = self.calculate_attention(q, k, v, i, kin)
            attn_values = attn_values.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)
            
            # Residual connection and dropout
            Z1 = state_i + self.dropout1(attn_values)
            
            # Layer normalization
            Z2 = self.norm1(Z1)
            
            # Extract mask for current timestep
            m_i = m[:,i,:,:].view(-1, self.patch_length, 1)
            
            # Update LSTM cell states
            C, M, H, N = self.LSTM(Z2, C, M, H, N, m_i)
            
            # Store attention weights
            A_list.append(A)
        
        # Stack attention matrices from all timesteps
        A_tensor = torch.stack(A_list, dim=1)
        
        return H, A_tensor

    def calculate_attention(self, q, k, v, t, kin):
        """Calculate attention scores and apply them to values.
        
        Args:
            q: Query tensor
            k: Key tensor
            v: Value tensor
            t: Current timestep
            kin: Additional modulation value
            
        Returns:
            tuple: (attention_applied_values, attention_weights)
        """
        # Calculate attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))
        
        # Apply softmax to get attention weights
        A = F.softmax(scores, dim=-1)
        
        # Modulate attention at specific timesteps
        if t >= 5:
            A = A*0
            A[:,:,:,0] = 0.9
            A[:,:,:,3] = 0.1
            
        # Apply attention weights to values
        return torch.matmul(A, v), A

class TransformerNetwork(nn.Module):
    """Main transformer network for visual attention processing.
    
    This network processes visual inputs through a transformer architecture
    to model visual attention and working memory.
    """
    def __init__(self, beta, input_dims=32, hidden_dim=128, fc1_dims=64, fc2_dims=32, n_actions=4,
                 name='transformer', chkpt_dir='td3_MAT'):
        super(TransformerNetwork, self).__init__()
        # Architecture dimensions
        self.input_dims = (128 + 4 + 8)  # VAE embedding + position encoding + temporal encoding
        self.hidden_dim = hidden_dim
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.patch_length = 4       # Number of spatial patches
        self.seq_length = 8         # Sequence length for temporal processing
        self.n_actions = n_actions
        
        # Checkpoint parameters
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = name + '_td3'
        
        # Transformer parameters
        self.d_model = self.input_dims
        self.num_heads = 1
        self.dff = 1024
        self.dropout = 0.01

        # Input projection and normalization
        # These layers process the raw input and generate an embedding
        self.fc_state_projection = nn.Linear(self.input_dims, self.input_dims)
        self.ln_state_projection = nn.LayerNorm(self.input_dims)
        self.fc2_state_projection = nn.Linear(self.input_dims, self.input_dims)
        self.ln2_state_projection = nn.LayerNorm(self.input_dims)

        # Transformer encoder block
        self.transformer_block1 = TransformerBlock(
            beta, 
            input_dims=128, 
            hidden_dim=self.hidden_dim, 
            fc1_dims=self.fc1_dims, 
            fc2_dims=self.fc2_dims, 
            n_actions=self.n_actions
        )

        self.optimizer = optim.Adam(self.parameters(), lr=beta, betas=(0.9, 0.99), eps=1e-8, weight_decay=1e-6)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)


    def forward(self, state, t, m, k):
        """Forward pass through the transformer network.
        
        Args:
            state: Input state tensor containing visual features
            t: Current timestep
            m: Mask tensor for conditional processing
            k: Attention modulation parameter
            
        Returns:
            tuple: (hidden_state, attention_weights)
        """
        # Reshape state to [batch, sequence, patch, features]
        state = state.view(-1, self.seq_length, self.patch_length, self.input_dims)
        batch_size = state.shape[0]
        
        # Flatten for initial processing
        state = state.view(batch_size * self.seq_length * self.patch_length, self.input_dims)
        
        # Generate embedding with residual connections
        # First projection with ELU activation
        X = F.elu(state + self.ln_state_projection(self.fc_state_projection(state)))
        # Second projection with layer normalization
        state = self.ln2_state_projection(state + self.fc2_state_projection(X))
        
        # Add small Gaussian noise for regularization
        noise = torch.randn_like(state) * 0.001
        state = state + noise
        
        # Reshape back to sequence format
        state = state.view(batch_size, self.seq_length, self.patch_length, self.input_dims)
        
        # Process through transformer block
        hidden_state, attention_weights = self.transformer_block1(state, m, k)
        
        return hidden_state, attention_weights

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