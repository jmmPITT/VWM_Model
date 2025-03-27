import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        # Encoder
        self.enc_conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=2, padding=1)
        self.enc_conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.enc_fc1 = nn.Linear(in_features=7*7*32, out_features=128)
        self.enc_fc2_mean = nn.Linear(in_features=128, out_features=20)  # Output layer for mean
        self.enc_fc2_log_var = nn.Linear(in_features=128, out_features=20)  # Output layer for log variance

        # Decoder
        self.dec_fc1 = nn.Linear(in_features=20, out_features=128)
        self.dec_fc2 = nn.Linear(in_features=128, out_features=7*7*32)
        self.dec_conv1 = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, stride=2, padding=1, output_padding=0)
        self.dec_conv2 = nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size=3, stride=2, padding=1, output_padding=0)

    def encode(self, x):
        # Encoder network
        # Permute x from [batch, height, width, channels] to [batch, channels, height, width]
        x = x.permute(0, 3, 1, 2)
        batch_size = x.size(0)
        x = F.relu(self.enc_conv1(x))
        x = F.relu(self.enc_conv2(x))
        # print(x.shape)
        x = x.reshape(batch_size, -1)  # Flatten the convolutional layer output
        # print(x.shape)
        x = F.relu(self.enc_fc1(x))
        # print(x.shape)
        return x  # Return mean and log variance

    def reparameterize(self, mu, log_var):
        # Reparameterization trick to sample from N(mu, var) where var = exp(log_var)
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        # Decoder network
        z = F.relu(self.dec_fc1(z))
        z = F.relu(self.dec_fc2(z))
        z = z.view(-1, 32, 7, 7)  # Reshape to feed into the transposed convolutional layers
        z = F.relu(self.dec_conv1(z))
        z = F.tanh(self.dec_conv2(z))
        # print('z',z.shape)
        return z  # Output layer with sigmoid activation to ensure output values are between 0 and 1

    def forward(self, x):
        # Forward pass through the entire VAE
        x = self.encode(x)
        # z = self.reparameterize(mu, log_var)
        return x
