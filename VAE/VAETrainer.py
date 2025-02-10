import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn.functional as F
import os

class CustomDataset(Dataset):
    def __init__(self, noisy_images, clean_images):
        self.noisy_images = torch.tensor(noisy_images, dtype=torch.float32)
        self.clean_images = torch.tensor(clean_images, dtype=torch.float32)

    def __len__(self):
        return len(self.noisy_images)

    def __getitem__(self, idx):
        return self.noisy_images[idx], self.clean_images[idx]

class VAERunner:
    def __init__(self, model, noisy_train_data, clean_train_data, batch_size=64, learning_rate=1e-3):
        self.model = model
        self.batch_size = batch_size

        # Convert numpy arrays to a custom PyTorch dataset
        self.train_dataset = CustomDataset(noisy_train_data, clean_train_data)

        # Create data loader
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def loss_function(self, recon_x, x, mu, logvar):
        # Reconstruction loss (MSE or BCE depending on the data)
        x = x.permute(0, 3, 1, 2)
        recon_loss = F.mse_loss(recon_x, x, reduction='sum')

        # KL divergence
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + kl_div

    def train(self, epochs):
        self.model.test()
        for epoch in range(epochs):
            overall_loss = 0
            for batch_idx, (noisy, clean) in enumerate(self.train_loader):
                noisy, clean = noisy.to(self.device), clean.to(self.device)
                self.optimizer.zero_grad()
                recon_batch, mu, logvar = self.model(noisy)
                loss = self.loss_function(recon_batch, clean, mu, logvar)
                loss.backward()
                overall_loss += loss.item()
                self.optimizer.step()

            print(f"Epoch [{epoch+1}/{epochs}], Loss: {overall_loss / len(self.train_loader.dataset):.4f}")

            if epoch % 10 == 0:
                # epoch_save_path = os.path.join(save_model_path, f'epoch_{epoch + 1}.pth')
                self.save_model()

    def save_model(self, file_path='vae_model.pth'):
        """
        Saves the model's state dictionary to a specified file path.

        Args:
        - file_path (str): Path where the model state dictionary should be saved.
        """
        torch.save(self.model.state_dict(), file_path)
        print(f"Model saved to {file_path}")
