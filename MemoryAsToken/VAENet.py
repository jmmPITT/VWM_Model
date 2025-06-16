import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
import h5py

# ---------------------------------------------------------------------------
# Example: How to load data using h5py (for reference)
# ---------------------------------------------------------------------------
# # Open the HDF5 file in read mode
# with h5py.File('BigData.h5', 'r') as h5f:
#     # Load a specific dataset from the HDF5 file
#     data = h5f['noise=PixelSwap/ChangeDegree=(0,35)'][:]
#
# # Reshape the data for model input (e.g., adding a channel dimension)
# data = data.reshape((-1, 25, 25, 1))
# print(f"Loaded data shape: {data.shape}")
#
# ---------------------------------------------------------------------------

class VAE(nn.Module):
    """
    A Variational Autoencoder (VAE) architecture used as a feature extractor.

    This class defines the layers for a VAE but is specifically implemented
    to use only the encoder portion to generate a fixed-size feature vector
    (embedding) from an input image. The decoder part is defined but not
    used in the primary `forward` pass.
    """
    def __init__(self):
        super(VAE, self).__init__()

        # --- Encoder Layers ---
        # These layers transform an input image into a 128-dimensional feature vector.
        self.enc_conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=2, padding=1)
        self.enc_conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.enc_fc1 = nn.Linear(in_features=7*7*32, out_features=128)

        # Note: These layers are defined for a complete VAE but are not used in this implementation's
        # forward pass. They would typically produce the parameters of the latent distribution.
        self.enc_fc2_mean = nn.Linear(in_features=128, out_features=20)
        self.enc_fc2_log_var = nn.Linear(in_features=128, out_features=20)

        # --- Decoder Layers ---
        # These layers would be used to reconstruct an image from a latent vector (z).
        # They are included for completeness of the VAE architecture but are not called.
        self.dec_fc1 = nn.Linear(in_features=20, out_features=128)
        self.dec_fc2 = nn.Linear(in_features=128, out_features=7*7*32)
        self.dec_conv1 = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, stride=2, padding=1, output_padding=0)
        self.dec_conv2 = nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size=3, stride=2, padding=1, output_padding=0)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Processes an input tensor through the convolutional and initial fully-connected
        layers to produce a 128-dimensional feature embedding.

        Args:
            x (torch.Tensor): The input image tensor of shape [batch, height, width, channels].

        Returns:
            torch.Tensor: A feature vector of shape [batch, 128].
        """
        # Permute from [batch, H, W, C] to PyTorch's required [batch, C, H, W]
        x = x.permute(0, 3, 1, 2)
        batch_size = x.size(0)

        # Pass through convolutional layers
        x = F.relu(self.enc_conv1(x))
        x = F.relu(self.enc_conv2(x))

        # Flatten the feature map to a vector.
        # Use .reshape() for robustness with non-contiguous tensors from .permute().
        x = x.reshape(batch_size, -1)

        # Pass through the first fully-connected layer to get the final embedding
        x = F.relu(self.enc_fc1(x))
        return x

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """
        Performs the reparameterization trick. (Defined for VAE completeness).
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decodes a latent vector back into an image. (Defined for VAE completeness).
        """
        z = F.relu(self.dec_fc1(z))
        z = F.relu(self.dec_fc2(z))
        z = z.view(-1, 32, 7, 7)
        z = F.relu(self.dec_conv1(z))
        z = F.sigmoid(self.dec_conv2(z))
        return z

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        The main forward pass for using the VAE as a feature extractor.

        This method calls the encoder to generate a feature vector and returns it directly.
        It does not perform reparameterization or decoding.

        Args:
            x (torch.Tensor): The input image tensor.

        Returns:
            torch.Tensor: The 128-dimensional feature vector from the encoder.
        """
        embedding = self.encode(x)
        return embedding
