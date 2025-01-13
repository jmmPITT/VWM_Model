########################################
# tester.py
########################################

import torch
import numpy as np
from torch.utils.data import DataLoader
import torch.nn as nn

from Network import TransformerNetwork
from trainer import ChangeDetectionDataset  # or define again

class Tester:
    def __init__(self,
                 model,
                 data_path='trial_observations_test.npy',
                 labels_path='trial_labels_test.npy',
                 batch_size=32,
                 num_epochs=1,
                 shuffle=False):
        """
        For evaluating performance on test data.
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

        self.criterion = nn.MSELoss()
        self.device = self.model.device

    def test(self):
        self.model.eval()  # Put model in eval mode
        total_loss = 0.0
        correct = 0
        total_samples = 0
        dff = 1024
        patch_num = 4
        patch_dim = 140
        T_end = 7


        with torch.no_grad():  # No gradient updates
            for batch_idx, (observations, labels) in enumerate(self.dataloader):
                # Move to device
                observations = observations.to(self.device)
                labels = labels.to(self.device)

                batch_size = observations.shape[0]

                # Re-initialize your hidden states, etc.
                C = torch.zeros(batch_size, patch_num, dff).to(self.device)
                M = torch.zeros(batch_size, patch_num, dff).to(self.device)
                H = torch.zeros(batch_size, patch_num, dff).to(self.device)
                N = torch.zeros(batch_size, patch_num, dff).to(self.device)

                # Evaluate across T_end steps
                step_loss = 0.0
                for i in range(T_end):  # T_end
                    obs_i = observations[:, i, :, :].view(-1, patch_num, patch_dim)
                    lab_i = labels[:, i, 0].view(-1, 1)

                    outputs, C, M, H, N, A = self.model(obs_i, C, M, H, N)
                    loss = self.criterion(outputs, lab_i)
                    step_loss += loss

                    # Evaluate accuracy only on the final step (or all steps if you prefer)
                    if i == T_end-1:
                        predicted = (outputs >= 0.5).float()
                        correct += (predicted == lab_i).sum().item()
                        total_samples += lab_i.size(0)

                total_loss += step_loss.item()

        avg_loss = total_loss / len(self.dataloader)
        accuracy = 100.0 * correct / total_samples if total_samples > 0 else 0.0

        print(f"[TEST] Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2f}%")
        return accuracy