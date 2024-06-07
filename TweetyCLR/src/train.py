'''
This training object should take hyperparameters necessary for training:
1. Learning rate
2. Number of epochs
3. Batch size
3. Optimizer
4. Loss function
5. train dataloader
6. test dataloader
'''

from model import Encoder
import torch
from tqdm import tqdm
from utils import moving_average
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class Trainer:
    def __init__(self, model, train_dataloader, test_dataloader, loss_fn, optimizer, device, params, lr = 0.001, num_epochs = 100, batch_size = 64, temperature = 1):
        self.mdl = model.to(device)
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        self.params = params
        self.lr = lr
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        # self.batch_means_train = []
        # self.batch_vars_train = []
        # self.batch_means_val = []
        # self.batch_vars_val = []
        # self.running_means_train = []
        # self.running_vars_train = []
        # self.running_means_val = []
        # self.running_vars_val = []

        # Lists to store losses
        self.train_losses = []
        self.test_losses = []

        self.mdl = self.mdl.to(torch.float32)
        self.tau = temperature

    def train(self):
        '''
        This function should run a training iteration of the model
        '''
        
        for epoch in range(self.num_epochs):
            print(f"Epoch [{epoch+1}/{self.num_epochs}]")
            
            # Training phase
            self.mdl.train()
            running_train_loss = 0.0
            running_test_loss = 0.0
            for batch_idx, (batch, labels) in enumerate(tqdm(self.train_dataloader, desc="Processing Batches")):
                actual_batch_size = batch.size(0)
                if actual_batch_size < self.batch_size * 2:
                    # If the actual batch size is less than double the expected batch size,
                    # we cannot create full original and augmented pairs
                    continue
                
                batch = batch.to(torch.float32)
                # Forward pass
                # Compute loss
                # Separate the original and augmented data
                original_data = batch[:self.batch_size]  # Shape: (64, 1, 100, 151)
                augmented_data = batch[self.batch_size:]  # Shape: (64, 1, 100, 151)

                # Compute embeddings for the original and augmented data
                original_embeddings = self.mdl(original_data)  # Shape: (64, embedding_dim)
                augmented_embeddings = self.mdl(augmented_data)  # Shape: (64, embedding_dim)

                # Normalize embeddings
                original_embeddings = F.normalize(original_embeddings, dim=1)
                augmented_embeddings = F.normalize(augmented_embeddings, dim=1)

                # Compute similarities (dot products) between original and augmented embeddings
                similarity_matrix = torch.mm(original_embeddings, augmented_embeddings.T)  # Shape: (64, 64)
                similarity_matrix /= self.tau


                # Create labels (positive pairs are diagonal elements)
                labels = torch.arange(self.batch_size).long().to(batch.device)

                # Compute the InfoNCE loss
                loss = F.cross_entropy(similarity_matrix, labels)






                # loss = self.loss_fn(ref=anchor_rep, pos=positive_rep, neg=negative_rep)
                
                # Backward pass and optimization
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Store training loss
                self.train_losses.append(loss.item())
                running_train_loss += loss.item()

                # # Compute batch statistics for training
                # batch_mean, batch_var = self.mdl.get_batch_stats(anchors)
                # self.batch_means_train.append(batch_mean.cpu().numpy())
                # self.batch_vars_train.append(batch_var.cpu().numpy())

                # Get running statistics for training
                # running_means, running_vars = self.mdl.get_running_stats()
                # self.running_means_train.append(running_means)
                # self.running_vars_train.append(running_vars)

                # Evaluation phase (on the current batch from the test loader)
                self.mdl.eval()
                with torch.no_grad():
                    try:
                        test_batch = next(iter(self.test_dataloader))
                        batch, labels = test_batch

                        actual_batch_size = batch.size(0)
                        if actual_batch_size < self.batch_size * 2:
                            # If the actual batch size is less than double the expected batch size,
                            # we cannot create full original and augmented pairs
                            continue

                        batch = batch.to(torch.float32)

                        # Compute loss
                        # Separate the original and augmented data
                        original_data = batch[:self.batch_size]  # Shape: (64, 1, 100, 151)
                        augmented_data = batch[self.batch_size:]  # Shape: (64, 1, 100, 151)

                        # Compute embeddings for the original and augmented data
                        original_embeddings = self.mdl(original_data)  # Shape: (64, embedding_dim)
                        augmented_embeddings = self.mdl(augmented_data)  # Shape: (64, embedding_dim)

                        # Normalize embeddings
                        original_embeddings = F.normalize(original_embeddings, dim=1)
                        augmented_embeddings = F.normalize(augmented_embeddings, dim=1)

                        # Compute similarities (dot products) between original and augmented embeddings
                        similarity_matrix = torch.mm(original_embeddings, augmented_embeddings.T)  # Shape: (64, 64)
                        similarity_matrix /= self.tau

                        # Create labels (positive pairs are diagonal elements)
                        labels = torch.arange(self.batch_size).long().to(batch.device)

                        # Compute the InfoNCE loss
                        test_loss = F.cross_entropy(similarity_matrix, labels)

                        # Store testing loss
                        self.test_losses.append(test_loss.item())
                        running_test_loss += test_loss.item()

                        # Compute batch statistics for validation
                        # batch_mean, batch_var = self.mdl.get_batch_stats(test_anchors)
                        # self.batch_means_val.append(batch_mean.cpu().numpy())
                        # self.batch_vars_val.append(batch_var.cpu().numpy())

                        # Get running statistics for validation
                        # running_means, running_vars = self.mdl.get_running_stats()
                        # self.running_means_val.append(running_means)
                        # self.running_vars_val.append(running_vars)
                    except StopIteration:
                        continue
                self.mdl.train()

                # Print average loss for every 100 batches
                if (batch_idx + 1) % 100 == 0:
                    avg_train_loss = running_train_loss / 100
                    avg_test_loss = running_test_loss / 100
                    print(f"Average Training Loss after {batch_idx + 1} batches: {avg_train_loss:.4f}")
                    print(f"Average Testing Loss after {batch_idx + 1} batches: {avg_test_loss:.4f}")
                    running_train_loss = 0.0
                    running_test_loss = 0.0

        self.plot_loss_curves()
        # self.plot_batch_stats()
        # self.plot_running_stats()

    def plot_loss_curves(self):
        def moving_average(x, w):
            return np.convolve(x, np.ones(w), 'valid') / w

        # Compute moving average with a window size of 10
        window_size = 10
        train_losses_ma = moving_average(self.train_losses, window_size)
        test_losses_ma = moving_average(self.test_losses, window_size)

        # Plot the losses and their moving averages
        plt.figure(figsize=(12, 6))
        plt.plot(self.train_losses, label='Training Loss', alpha=0.5)
        plt.plot(range(window_size-1, len(self.train_losses)), train_losses_ma, label='Training Loss (MA)')
        plt.plot(self.test_losses, label='Testing Loss', alpha=0.5)
        plt.plot(range(window_size-1, len(self.test_losses)), test_losses_ma, label='Testing Loss (MA)')
        plt.xlabel('Batch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training and Testing Losses with Moving Averages')
        plt.show()

    def plot_batch_stats(self):
        train_means = np.array(self.batch_means_train)
        train_vars = np.array(self.batch_vars_train)
        val_means = np.array(self.batch_means_val)
        val_vars = np.array(self.batch_vars_val)

        for i in range(train_means.shape[1]):  # Assuming channel dimension
            plt.figure(figsize=(12, 6))
            plt.plot(train_means[:, i], label=f'Train mean channel {i}')
            plt.plot(val_means[:, i], label=f'Validation mean channel {i}', linestyle='dashed')
            plt.xlabel('Batch')
            plt.ylabel('Mean')
            plt.legend()
            plt.title(f'Train vs Validation Mean for Channel {i}')
            plt.show()

            plt.figure(figsize=(12, 6))
            plt.plot(train_vars[:, i], label=f'Train variance channel {i}')
            plt.plot(val_vars[:, i], label=f'Validation variance channel {i}', linestyle='dashed')
            plt.xlabel('Batch')
            plt.ylabel('Variance')
            plt.legend()
            plt.title(f'Train vs Validation Variance for Channel {i}')
            plt.show()

    def plot_running_stats(self):
        def concatenate_running_stats(running_stats):
            # Ensure all running stats have consistent shapes
            flattened_stats = []
            for batch in running_stats:
                for stat in batch:
                    flattened_stats.append(stat)
            return np.array(flattened_stats)

        train_running_means = concatenate_running_stats(self.running_means_train)
        train_running_vars = concatenate_running_stats(self.running_vars_train)
        val_running_means = concatenate_running_stats(self.running_means_val)
        val_running_vars = concatenate_running_stats(self.running_vars_val)

        num_channels = train_running_means.shape[1]

        for i in range(num_channels):
            plt.figure(figsize=(12, 6))
            plt.plot(train_running_means[:, i], label=f'Train running mean channel {i}')
            plt.plot(val_running_means[:, i], label=f'Validation running mean channel {i}', linestyle='dashed')
            plt.xlabel('Batch')
            plt.ylabel('Running Mean')
            plt.legend()
            plt.title(f'Train vs Validation Running Mean for Channel {i}')
            plt.show()

            plt.figure(figsize=(12, 6))
            plt.plot(train_running_vars[:, i], label=f'Train running variance channel {i}')
            plt.plot(val_running_vars[:, i], label=f'Validation running variance channel {i}', linestyle='dashed')
            plt.xlabel('Batch')
            plt.ylabel('Running Variance')
            plt.legend()
            plt.title(f'Train vs Validation Running Variance for Channel {i}')
            plt.show()