import os
import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.utils.rnn as rnn_utils


class CustomLSTMCell(nn.Module):
    """
    Custom LSTM cell that integrates additional nonlinear transformations.

    This cell processes inputs that have been reshaped into patches. The cell computes
    gate activations using separate linear transformations for the hidden state and
    the current input. It then uses these activations to update the cell state and
    produce an output hidden state.

    Attributes:
        patch_size (int): Number of patches per sequence.
        d_model (int): Dimensionality of the model.
        dff (int): Dimensionality for the internal feedforward network (default 512).
        input_size (int): Expected size of the input vector.
    """

    def __init__(self, patch_size: int, d_model: int) -> None:
        super(CustomLSTMCell, self).__init__()
        self.patch_size = patch_size
        self.d_model = d_model
        self.dff = 512
        self.input_size = self.d_model

        # Linear transformations for input gate, forget gate, output gate, and candidate state
        self.WI = nn.Linear(self.dff, self.dff)
        self.WF = nn.Linear(self.dff, self.dff)
        self.WO = nn.Linear(self.dff, self.dff)
        self.WZ = nn.Linear(self.dff, self.dff)

        # Linear transformations for the current input (referred to as "recurrent" weights)
        self.RI = nn.Linear(self.input_size, self.dff)
        self.RF = nn.Linear(self.input_size, self.dff)
        self.RO = nn.Linear(self.input_size, self.dff)
        self.RZ = nn.Linear(self.input_size, self.dff)

    def forward(self, Zi: T.Tensor, Ci: T.Tensor, Mi: T.Tensor, Hi: T.Tensor, Ni: T.Tensor) -> tuple:
        """
        Forward pass for the custom LSTM cell.

        Args:
            Zi (T.Tensor): Input tensor of shape (batch, patch_size * input_size).
            Ci (T.Tensor): Previous cell state tensor.
            Mi (T.Tensor): Previous normalization term tensor.
            Hi (T.Tensor): Previous hidden state tensor.
            Ni (T.Tensor): Previous normalization scaling factor tensor.

        Returns:
            tuple: Updated cell state, normalization term, hidden state, and scaling factor.
        """
        # Reshape inputs to (batch, patch_size, feature_dim)
        Zi = Zi.view(-1, self.patch_size, self.input_size)
        Ci = Ci.view(-1, self.patch_size, self.dff)
        Hi = Hi.view(-1, self.patch_size, self.dff)
        Ni = Ni.view(-1, self.patch_size, self.dff)

        C_prev = Ci
        M_prev = Mi
        H_prev = Hi

        # Compute gate pre-activations from hidden state and current input
        I_tilde = self.WI(H_prev) + self.RI(Zi)
        F_tilde = self.WF(H_prev) + self.RF(Zi)
        O_tilde = self.WO(H_prev) + self.RO(Zi)
        Z_tilde = self.WZ(H_prev) + self.RZ(Zi)

        # Compute normalization term and gate activations with exponential normalization
        M_t = T.max(F_tilde + M_prev, I_tilde)
        I_t = T.exp(I_tilde - M_t)
        F_t = T.exp(F_tilde + M_prev - M_t)

        O_t = T.sigmoid(O_tilde)
        N_t = F_t * Ni + I_t
        Z_t = T.tanh(Z_tilde)

        # Update cell state and compute hidden state
        C_t = (C_prev * F_t + Z_t * I_t)
        H_t = O_t * (C_t / N_t)

        return C_t, M_t, H_t, N_t


class TransformerBlock(nn.Module):
    """
    Transformer block that integrates multi-head self-attention, layer normalization,
    and a custom LSTM cell.

    The block first projects the input into query, key, and value matrices, computes
    the attention weights, and then combines them to produce an attention output.
    This output is then combined with the original state (via residual connection) and
    normalized.
    """

    def __init__(self, beta: float, input_dims: int = 32, hidden_dim: int = 128,
                 fc1_dims: int = 64, fc2_dims: int = 32, n_actions: int = 4,
                 name: str = 'transformer', chkpt_dir: str = 'td3_MAT') -> None:
        super(TransformerBlock, self).__init__()
        # Overwrite input dimensions to fixed image dimensions (e.g., 30x30 RGB)
        self.input_dims = 30 * 30 * 3
        self.hidden_dim = hidden_dim
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.patch_length = 9  # Number of patches
        self.n_actions = n_actions
        self.d_model = self.input_dims  # Model dimension matches flattened image patch size
        self.num_heads = 1
        self.dff = 512
        self.dropout = 0.01

        # Calculate dimension per attention head
        self.d_k = self.d_model // self.num_heads

        # Query, key, and value linear transformations
        self.W_q = nn.Linear(self.d_model, self.d_k * self.num_heads)
        self.W_k = nn.Linear(self.d_model, self.d_k * self.num_heads)
        self.W_v = nn.Linear(self.d_model, self.d_k * self.num_heads)

        # Additional transformations applied to auxiliary hidden inputs H1 and H2
        self.W_C1q = nn.Linear(self.dff, self.d_k * self.num_heads)
        self.W_C1k = nn.Linear(self.dff, self.d_k * self.num_heads)
        self.W_C1v = nn.Linear(self.dff, self.d_k * self.num_heads)

        self.W_C2q = nn.Linear(self.dff, self.d_k * self.num_heads)
        self.W_C2k = nn.Linear(self.dff, self.d_k * self.num_heads)
        self.W_C2v = nn.Linear(self.dff, self.d_k * self.num_heads)

        # Position-wise feedforward layers
        self.linear1 = nn.Linear(self.num_heads * self.d_k, self.d_model * 4)
        self.linear2 = nn.Linear(self.d_model * 4, self.d_model)
        self.dropout1 = nn.Dropout(self.dropout)

        # Layer normalization and second dropout layer for residual connections
        self.norm1 = nn.LayerNorm(self.d_model)
        self.norm2 = nn.LayerNorm(self.d_model)
        self.dropout2 = nn.Dropout(self.dropout)

        # Custom LSTM cell (currently not actively used in the forward pass)
        self.LSTM = CustomLSTMCell(patch_size=self.patch_length, d_model=self.d_model)

        # Define an optimizer (if training within the block) and set device
        self.optimizer = optim.Adam(self.parameters(), lr=beta, betas=(0.9, 0.99),
                                    eps=1e-8, weight_decay=1e-6)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state: T.Tensor, H1: T.Tensor, H2: T.Tensor) -> T.Tensor:
        """
        Forward pass for the transformer block.

        Args:
            state (T.Tensor): Input state tensor of shape (batch, patch_length * input_dims).
            H1 (T.Tensor): Auxiliary hidden state tensor 1.
            H2 (T.Tensor): Auxiliary hidden state tensor 2.

        Returns:
            T.Tensor: Processed state tensor after attention and residual connection.
        """
        # Reshape state into patches: (batch, patch_length, input_dims)
        state = state.view(-1, self.patch_length, self.input_dims)
        batch_size = state.shape[0]

        # Construct query, key, and value matrices and combine with auxiliary inputs
        q = self.W_q(state).view(batch_size, -1, self.num_heads, self.d_k) * (
                self.W_C1q(H1).view(batch_size, -1, self.num_heads, self.d_k) +
                self.W_C2q(H2).view(batch_size, -1, self.num_heads, self.d_k))
        k = self.W_k(state).view(batch_size, -1, self.num_heads, self.d_k) * (
                self.W_C1k(H1).view(batch_size, -1, self.num_heads, self.d_k) +
                self.W_C2k(H2).view(batch_size, -1, self.num_heads, self.d_k))
        v = self.W_v(state).view(batch_size, -1, self.num_heads, self.d_k) * (
                self.W_C1v(H1).view(batch_size, -1, self.num_heads, self.d_k) +
                self.W_C2v(H2).view(batch_size, -1, self.num_heads, self.d_k))

        # Transpose to shape (batch, num_heads, seq_len, d_k)
        q, k, v = [x.transpose(1, 2) for x in (q, k, v)]

        # Compute scaled dot-product attention
        attn_values, _ = self.calculate_attention(q, k, v)
        attn_values = attn_values.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)

        # Apply dropout and add residual connection, followed by layer normalization
        Z1 = state + self.dropout1(attn_values)
        Z2 = self.norm1(Z1)

        # The commented-out code below indicates additional feedforward layers or LSTM usage if desired
        # Z3 = F.gelu(self.linear1(Z2))
        # Z4 = self.linear2(Z3)
        # Z5 = Z2 + self.dropout2(Z4)
        # C, M, H, N = self.LSTM(Z2, C, M, H, N)

        return Z2

    def calculate_attention(self, q: T.Tensor, k: T.Tensor, v: T.Tensor) -> tuple:
        """
        Compute the attention values using scaled dot-product attention.

        Args:
            q (T.Tensor): Query tensor.
            k (T.Tensor): Key tensor.
            v (T.Tensor): Value tensor.

        Returns:
            tuple: The attention output and the attention weights.
        """
        # Compute attention scores and scale them
        scores = T.matmul(q, k.transpose(-2, -1)) / T.sqrt(T.tensor(self.d_k, dtype=T.float32))
        A = F.softmax(scores, dim=-1)
        return T.matmul(A, v), A


class TransformerNetwork(nn.Module):
    """
    Transformer-based network that combines transformer blocks and custom LSTM cells.

    This network processes an input state through a transformer block and then updates
    internal hidden states via a series of custom LSTM cells. The output includes updated
    internal states and a transformed representation.
    """

    def __init__(self, beta: float, input_dims: int = 32, hidden_dim: int = 128,
                 fc1_dims: int = 64, fc2_dims: int = 32, n_actions: int = 4,
                 name: str = 'transformer', chkpt_dir: str = 'td3_MAT') -> None:
        super(TransformerNetwork, self).__init__()
        self.input_dims = 30 * 30 * 3  # Fixed flattened image dimensions
        self.hidden_dim = hidden_dim
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.patch_length = 9  # Number of patches
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = name + '_td3'
        self.d_model = self.input_dims
        self.num_heads = 1
        self.dff = 512
        self.dropout = 0.01

        # Initialize two custom LSTM cells for sequential processing
        self.LSTM1 = CustomLSTMCell(patch_size=self.patch_length, d_model=self.input_dims)
        self.LSTM2 = CustomLSTMCell(patch_size=self.patch_length, d_model=self.dff)

        # Initialize a transformer block to process the state
        self.transformer_block1 = TransformerBlock(beta, input_dims=self.input_dims,
                                                   hidden_dim=self.hidden_dim,
                                                   fc1_dims=self.fc1_dims, fc2_dims=self.fc2_dims,
                                                   n_actions=self.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=beta, betas=(0.9, 0.99),
                                    eps=1e-8, weight_decay=1e-6)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state: T.Tensor,
                C1: T.Tensor, M1: T.Tensor, H1: T.Tensor, N1: T.Tensor,
                C2: T.Tensor, M2: T.Tensor, H2: T.Tensor, N2: T.Tensor) -> tuple:
        """
        Forward pass for the transformer network.

        Args:
            state (T.Tensor): Input state tensor.
            C1, M1, H1, N1: Internal states for the first LSTM cell.
            C2, M2, H2, N2: Internal states for the second LSTM cell.

        Returns:
            tuple: Updated internal states and the transformed representation.
        """
        # Reshape state to include a dummy dimension for consistency with patch_length
        state = state.view(-1, 1, self.patch_length, self.input_dims)
        batch_size = state.shape[0]

        # Process state through the transformer block using auxiliary hidden states H1 and H2
        Z = self.transformer_block1(state, H1, H2)

        # Update first set of internal states via LSTM1 using transformer output Z
        C1, M1, H1, N1 = self.LSTM1(Z, C1, M1, H1, N1)

        # Update second set of internal states via LSTM2 using the updated hidden state H1
        C2, M2, H2, N2 = self.LSTM2(H1, C2, M2, H2, N2)

        return C1, M1, H1, N1, C2, M2, H2, N2, Z

    def save_checkpoint(self) -> None:
        """Save the current model parameters to a checkpoint file."""
        print('... saving checkpoint ...')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self) -> None:
        """Load model parameters from a checkpoint file."""
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.checkpoint_file))


class CriticNetwork(nn.Module):
    """
    Critic network for the distributional actor-critic framework.

    This network combines the transformed state representation from the transformer
    with an action embedding to produce Q-value distributions over actions.
    """

    def __init__(self, beta: float, input_dims: int = 32, hidden_dim: int = 512,
                 fc1_dims: int = 256, fc2_dims: int = 128, n_actions: int = 4,
                 name: str = 'critic', chkpt_dir: str = 'td3_MAT') -> None:
        super(CriticNetwork, self).__init__()
        # Fixed input dimensions after transformer processing (e.g., feature dimension 512)
        self.input_dims = 512
        self.hidden_dim = hidden_dim
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.patch_length = 9
        self.seq_length = 1
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = name + '_td3'

        # Project action vector to match the flattened transformer state dimensions
        self.fc_action_lever_projection = nn.Linear(self.n_actions, self.input_dims * self.patch_length)
        self.ln_action_lever_projection = nn.LayerNorm(self.input_dims * self.patch_length)

        # Combine transformer state and projected action, then pass through fully connected layers
        self.fc1 = nn.Linear(self.input_dims * self.patch_length + self.input_dims * self.patch_length, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, fc1_dims)
        self.ln2 = nn.LayerNorm(fc1_dims)
        self.fc3 = nn.Linear(fc1_dims, fc2_dims)
        self.ln3 = nn.LayerNorm(fc2_dims)

        # Final output layer producing Q-value distribution (using softmax)
        self.q = nn.Linear(fc2_dims, 30)

        self.optimizer = optim.Adam(self.parameters(), lr=beta, betas=(0.9, 0.99),
                                    eps=1e-8, weight_decay=1e-6)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, transformer_state: T.Tensor, action: T.Tensor) -> T.Tensor:
        """
        Forward pass for the critic network.

        Args:
            transformer_state (T.Tensor): Transformed state representation.
            action (T.Tensor): One-hot encoded action vector.

        Returns:
            T.Tensor: Q-value distribution over actions.
        """
        # Flatten transformer state for concatenation
        transformer_state = transformer_state.view(-1, self.patch_length * self.input_dims)
        # Process action vector through projection layers
        action = action.view(-1, self.n_actions)
        action_lever = self.fc_action_lever_projection(action)
        # Concatenate state and action projections
        state_action = T.cat((transformer_state, action_lever), dim=1)

        x = F.elu(self.ln1(self.fc1(state_action)))
        x = F.elu(self.ln2(self.fc2(x)))
        x = F.elu(self.ln3(self.fc3(x)))
        q = self.q(x)
        q = F.softmax(q, dim=1)
        return q

    def save_checkpoint(self) -> None:
        """Save the critic network parameters to a checkpoint file."""
        print('... saving checkpoint ...')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self) -> None:
        """Load critic network parameters from a checkpoint file."""
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.checkpoint_file))


class ActorNetwork(nn.Module):
    """
    Actor network for the distributional actor-critic framework.

    This network produces a policy distribution over actions given the transformed state
    representation. Noise is added to the logits for exploration.
    """

    def __init__(self, alpha: float, input_dims: int = 16, hidden_dim: int = 512,
                 fc1_dims: int = 256, fc2_dims: int = 128, n_actions: int = 4,
                 name: str = 'Actor', chkpt_dir: str = 'td3_MAT') -> None:
        super(ActorNetwork, self).__init__()
        self.input_dims = 512
        self.hidden_dim = hidden_dim
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.patch_length = 9
        self.seq_length = 1
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = name + '_td3'

        # Fully connected layers for processing the transformer state
        self.fc1 = nn.Linear(self.input_dims * self.patch_length, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, fc1_dims)
        self.ln2 = nn.LayerNorm(fc1_dims)
        self.fc3 = nn.Linear(fc1_dims, fc2_dims)
        self.ln3 = nn.LayerNorm(fc2_dims)
        # Final layer to produce logits for action probabilities
        self.pi_lever = nn.Linear(fc2_dims, self.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha, betas=(0.9, 0.99),
                                    eps=1e-8, weight_decay=1e-6)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, transformer_state: T.Tensor) -> tuple:
        """
        Forward pass for the actor network.

        Args:
            transformer_state (T.Tensor): Transformed state representation.

        Returns:
            tuple: A tuple containing the policy distribution (after softmax) and the raw logits.
        """
        # Flatten transformer state for processing
        transformer_state = transformer_state.view(-1, self.patch_length * self.input_dims)
        x1 = F.elu(self.ln1(self.fc1(transformer_state)))
        x2 = F.elu(self.ln2(self.fc2(x1)))
        x3 = F.elu(self.ln3(self.fc3(x2)))
        logits = self.pi_lever(x3)

        # Add a small amount of noise for exploration
        noise_scale = 0.0001
        noise = T.randn_like(logits) * noise_scale
        noisy_logits = logits + noise

        # Compute action probabilities using softmax
        pi_lever = F.softmax(noisy_logits, dim=-1)
        return pi_lever, logits

    def save_checkpoint(self) -> None:
        """Save the actor network parameters to a checkpoint file."""
        print('... saving checkpoint ...')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self) -> None:
        """Load actor network parameters from a checkpoint file."""
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.checkpoint_file))
