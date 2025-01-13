import torch
from torch.utils.data import Dataset
import numpy as np
from Network import *


class ChangeDetectionDataset(Dataset):
    def __init__(self, data_path='trial_observation_data.npy',
                 labels_path='trial_labels.npy', transform=None):
        """
        Args:
            data_path: Path to .npy file of shape (n_games, T_end, 4, 25^2)
            labels_path: Path to .npy file of shape (n_games, 1)
            transform: Optional transform to apply to each observation
        """
        super().__init__()
        # Load data from disk
        self.data = np.load(data_path)  # shape: (n_games, T_end, 4, 25^2)
        print(self.data.shape)
        self.labels = np.load(labels_path)  # shape: (n_games, 1)

        # Convert to float32 and torch tensors
        self.data = torch.from_numpy(self.data).float()
        self.labels = torch.from_numpy(self.labels).float()

        self.transform = transform

    def __len__(self):
        return len(self.data)  # n_games

    def __getitem__(self, idx):
        """
        Returns:
            observation: shape (T_end * 4, 25^2)
            label: shape (1,)
        """
        observation = self.data[idx]  # shape: (T_end, 4, 25^2)
        label = self.labels[idx]  # shape: (1,)

        # Reshape to (T_end*4, 25^2) so it matches your transformer's expected shape
        observation = observation.view(7, 4, 128+4+8)  # shape: (T_end*4, 25^2)
        # print('observation.shape',observation.shape)
        if self.transform:
            observation = self.transform(observation)

        return observation, label


import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class Trainer:
    def __init__(self,
                 model,
                 data_path='trial_observations.npy',
                 labels_path='trial_labels.npy',
                 batch_size=32,
                 num_epochs=10,
                 shuffle=True):
        """
        Args:
            model: instance of your TransformerNetwork
            data_path: path to the data .npy
            labels_path: path to the labels .npy
            batch_size: number of samples in each batch
            num_epochs: number of epochs to train
            shuffle: whether to shuffle the dataset each epoch
        """
        self.model = model
        self.batch_size = batch_size
        self.num_epochs = num_epochs

        # Create dataset and dataloader
        self.dataset = ChangeDetectionDataset(data_path=data_path,
                                              labels_path=labels_path)

        self.dataloader = DataLoader(self.dataset,
                                     batch_size=self.batch_size,
                                     shuffle=shuffle)

        # Loss function for binary classification
        self.criterion = nn.MSELoss()

        # If you want to use the model's internal optimizer:
        # self.optimizer = model.optimizer

        # Or define an optimizer externally:
        # self.optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        self.device = self.model.device  # e.g. cuda:0 or cpu

    def train(self):
        for epoch in range(self.num_epochs):
            epoch_loss = 0.0
            correct = 0
            total = 0
            dff = 1024
            patch_num = 4
            patch_dim = 140
            T_end = 7

            for batch_idx, (observations, labels) in enumerate(self.dataloader):
                # observations shape: (batch_size, T_end*4, 25^2)
                # labels shape: (batch_size, 1)

                # Move to device
                observations = observations.to(self.device)
                # print('observations',observations.shape)
                labels = labels.to(self.device)
                # print('labels',labels.shape)

                batch_size = observations.shape[0]

                # Zero gradients
                self.model.optimizer.zero_grad()
                total_loss = 0

                C = torch.zeros(batch_size, patch_num, dff).to(
                    self.device)
                M = torch.zeros(batch_size, patch_num, dff).to(
                    self.device)
                H = torch.zeros(batch_size, patch_num, dff).to(
                    self.device)
                N = torch.zeros(batch_size, patch_num, dff).to(
                    self.device)


                for i in range(T_end):
                    obs_i = observations[:,i,:,:].view(-1,patch_num,patch_dim)
                    lab_i = labels[:, i, 0].view(-1,1)
                    # Forward pass
                    outputs, C, M, H, N, A  = self.model(obs_i, C, M, H, N )  # shape: (batch_size, 1)

                    # Compute loss
                    loss = self.criterion(outputs, lab_i)
                    total_loss += loss

                    # Round the outputs for binary classification, compute accuracy
                    predicted = (outputs >= 0.5).float()
                    correct += (predicted == lab_i).sum().item()
                    total += labels.size(0)

                # Backprop
                total_loss.backward()
                self.model.optimizer.step()

                # Update epoch metrics
                epoch_loss += total_loss.item()

                # Round the outputs for binary classification, compute accuracy
                # predicted = (outputs >= 0.5).float()
                # correct += (predicted == labels).sum().item()
                # total += labels.size(0)

            avg_loss = epoch_loss / len(self.dataloader)
            accuracy = 100.0 * correct / (total)

            print(f"Epoch [{epoch + 1}/{self.num_epochs}]  Loss: {avg_loss:.4f}  Accuracy: {accuracy:.2f}%")
            if (epoch + 1) % 10 == 0:
                checkpoint_path = "model_save.pth"
                # Save state_dict for both model and optimizer
                checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.model.optimizer.state_dict(),
                    'loss': avg_loss
                }
                torch.save(checkpoint, checkpoint_path)
                print(f"Checkpoint saved at {checkpoint_path}")
