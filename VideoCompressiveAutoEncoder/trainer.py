import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

# Import the TransformerNetwork class from your neural network file.
from network_sensor2 import TransformerNetwork


# ---------------------------
# Custom Dataset for Video Data
# ---------------------------
class MovingMNISTDataset(Dataset):
    def __init__(self, npy_file):
        # Load the data from the .npy file.
        self.data = np.load(npy_file)
        print('Dataset shape:', self.data.shape)

        # Create patched data: (num_frames, num_sequences, patches, height, width, channels)
        self.patched_data = np.zeros((330, 50, 256, 15, 20, 3), dtype=np.float32)
        c = 0
        for i in range(16):
            for j in range(16):
                # Extract spatial patches from the video frames.
                self.patched_data[:, :, c, :, :, :] = self.data[100:430, :, 15 * i:15 * (i + 1), 20 * j:20 * (j + 1), :]
                c += 1
        # Reshape and normalize to [0,1]
        self.patched_data = self.patched_data.reshape(-1, 50, 256, 15, 20, 3) / 255.0

    def __len__(self):
        return self.patched_data.shape[0]

    def __getitem__(self, idx):
        # Return a single video sequence.
        frame = self.patched_data[idx]
        return torch.from_numpy(frame)


# ---------------------------
# Trainer Class for the Temporal Autoencoder
# ---------------------------
class TemporalAutoencoderTrainer:
    def __init__(self, transformer_model, dataset, batch_size=16, learning_rate=1e-3, num_epochs=10):
        self.transformer_model = transformer_model
        self.dataset = dataset
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs

        # Set device.
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.transformer_model.to(self.device)

        # Create the optimizer.
        self.optimizer_TAE = optim.Adam(self.transformer_model.parameters(), lr=self.learning_rate)

        # Create the DataLoader.
        self.dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

    def train(self):
        for epoch in range(1, self.num_epochs + 1):
            epoch_loss = 0.0
            for batch_idx, sequences in enumerate(self.dataloader):
                # sequences: (batch_size, num_frames, num_patches, height, width, channels)
                sequences = sequences.to(self.device)
                batch_size, num_frames, num_patches, height, width, channels = sequences.size()

                # Build latent sequences by flattening each patch (height, width, channels) into one vector.
                latent_sequences = []
                with torch.no_grad():
                    for t in range(num_frames):
                        # Shape: (batch_size, num_patches, 15*20*3)
                        frames = sequences[:, t, :, :, :].view(batch_size, num_patches, -1)
                        latent_sequences.append(frames.unsqueeze(1))  # add time dimension

                # Concatenate along time: (batch_size, num_frames, num_patches, latent_dim)
                latent_sequences = torch.cat(latent_sequences, dim=1)
                # Reshape to match the network’s expected input:
                # (batch_size, num_frames, patch_length, input_dim)
                state = latent_sequences.view(batch_size, num_frames,
                                              self.transformer_model.patch_length, -1)

                # Initialize recurrent states (for the three LSTM cells) with zeros.
                C1 = torch.zeros(batch_size, self.transformer_model.patch_length, self.transformer_model.dff,
                                 device=self.device)
                M1 = torch.zeros_like(C1)
                H1 = torch.zeros_like(C1)
                N1 = torch.zeros_like(C1)
                C2 = torch.zeros_like(C1)
                M2 = torch.zeros_like(C1)
                H2 = torch.zeros_like(C1)
                N2 = torch.zeros_like(C1)
                C3 = torch.zeros_like(C1)
                M3 = torch.zeros_like(C1)
                H3 = torch.zeros_like(C1)
                N3 = torch.zeros_like(C1)

                total_loss = 0.0

                # Process each time step sequentially.
                for t in range(num_frames):
                    # Prepare input for the transformer (shape: [batch_size, patch_length, input_dim]).
                    input_state = state[:, t, :].view(batch_size, self.transformer_model.patch_length, -1)

                    # Forward pass through the transformer network.
                    C1, M1, H1, N1, C2, M2, H2, N2, C3, M3, H3, N3, PredSequence = \
                        self.transformer_model.forward(input_state, C1, M1, H1, N1,
                                                       C2, M2, H2, N2,
                                                       C3, M3, H3, N3, t)
                    # At final time step, compute the reconstruction loss.
                    if t == num_frames - 1:
                        PredSequence = PredSequence.view(batch_size, t + 1, self.transformer_model.patch_length, -1)
                        z_target = state[:, :t + 1, :].reshape(batch_size, t + 1, self.transformer_model.patch_length,
                                                               -1)
                        mse_loss = F.mse_loss(PredSequence, z_target)
                        total_loss += mse_loss

                # Backpropagation.
                self.optimizer_TAE.zero_grad()
                total_loss.backward()
                self.optimizer_TAE.step()

                epoch_loss += total_loss.item() * batch_size

                if batch_idx % 10 == 0:
                    print(f'Epoch [{epoch}/{self.num_epochs}] Batch [{batch_idx}/{len(self.dataloader)}] '
                          f'TAELoss: {total_loss.item() / batch_size:.6f}')

            avg_loss = epoch_loss / len(self.dataset)
            print(f'====> Epoch: {epoch} Average loss: {avg_loss:.6f}')

            # Save a model checkpoint after each epoch.
            self.save_model_TAE(epoch)
            print('Model saved!')

    def save_model_TAE(self, epoch):
        model_save_path = 'transformer_model_Long10H.pth'
        torch.save(self.transformer_model.state_dict(), model_save_path)
        print(f'Model saved at {model_save_path}')

    def load_model(self, epoch):
        model_save_path = f'transformer_model_epoch_{epoch}.pth'
        self.transformer_model.load_state_dict(torch.load(model_save_path, map_location=self.device))
        self.transformer_model.to(self.device)
        self.transformer_model.eval()
        print(f'Model loaded from {model_save_path}')


# ---------------------------
# Main Entry Point
# ---------------------------
if __name__ == '__main__':
    # Path to the dataset (.npy file).
    npy_file = 'ucf101_subset_batch_4.npy'
    dataset = MovingMNISTDataset(npy_file)

    # Initialize the transformer model and load pre-trained weights if available.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transformer_model_path = 'transformer_model_Long10H.pth'
    transformer_model = TransformerNetwork(beta=1e-5).to(device)
    transformer_model.load_state_dict(torch.load(transformer_model_path, map_location=device))

    # Instantiate the trainer and start training.
    trainer = TemporalAutoencoderTrainer(
        transformer_model=transformer_model,
        dataset=dataset,
        batch_size=2,
        learning_rate=1e-5,
        num_epochs=100000  # Adjust as needed.
    )
    trainer.train()
