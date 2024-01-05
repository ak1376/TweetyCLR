#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 11:11:09 2023

UMAP analysis of canary song spectrogram slices. Instead of doing a UMAP analysis on the spectrogram slices, I am doing a UMAP on the slice labels. The hope is that 

@author: AnanyaKapoor
"""

import numpy as np
import torch
import sys
filepath = '/Users/AnanyaKapoor'
import os
# os.chdir('/Users/AnanyaKapoor/Downloads/TweetyCLR')
os.chdir('/Users/AnanyaKapoor/Dropbox (University of Oregon)/Kapoor_Ananya/01_Projects/01_b_Canary_SSL/TweetyCLR_End_to_End')
from util import Tweetyclr
import torch.nn as nn
import torch.nn.functional as F
import os
import pickle
from torch.utils.data import DataLoader, TensorDataset
import umap
import matplotlib.pyplot as plt
import torch.optim as optim
import itertools
import inspect
import torch.nn.init as init
import random
torch.manual_seed(295)
np.random.seed(295)
random.seed(295)

plt.rcParams.update({'font.size': 20})
plt.rcParams['figure.figsize'] = [15, 15]  # width and height should be in inches, e.g., [10, 6]

bird_dir = f'{filepath}/Dropbox (University of Oregon)/Kapoor_Ananya/01_Projects/01_a_Rotations/Gardner_Lab/Canary_Data/llb16_data_matrices/'
audio_files = bird_dir+'llb3_songs'
directory = bird_dir+ 'Python_Files'

analysis_path = f'{filepath}/Dropbox (University of Oregon)/Kapoor_Ananya/01_Projects/01_b_Canary_SSL/TweetyCLR_End_to_End/'
# analysis_path = '/home/akapoor/Dropbox (University of Oregon)/Kapoor_Ananya/01_Projects/01_b_Canary_SSL/TweetyCLR_Repo/'

# # Parameters we set
num_spec = 1
window_size = 100
stride = 10

# Define the folder name
folder_name = f'{analysis_path}Supervised_Task/Num_Spectrograms_{num_spec}_Window_Size_{window_size}_Stride_{stride}'


files = os.listdir(directory)
all_songs_data = [f'{directory}/{element}' for element in files if '.npz' in element] # Get the file paths of each numpy file from Yarden's data
all_songs_data.sort()

masking_freq_tuple = (500, 7000)
spec_dim_tuple = (window_size, 151)


with open(f'{filepath}/Dropbox (University of Oregon)/Kapoor_Ananya/01_Projects/01_b_Canary_SSL/Canary_SSL_Repo/InfoNCE_Num_Spectrograms_100_Window_Size_100_Stride_10/category_colors.pkl', 'rb') as file:
    category_colors = pickle.load(file)


# The below cell initializes an object that will store useful information for contrastive learning

# In[1]: Creating Dataset


simple_tweetyclr_experiment_1 = Tweetyclr(num_spec, window_size, stride, folder_name, all_songs_data, masking_freq_tuple, spec_dim_tuple, category_colors)
# simple_tweetyclr_experiment_1.temperature_value = temp_value
simple_tweetyclr = simple_tweetyclr_experiment_1
simple_tweetyclr.first_time_analysis()

# In[2]: UMAP on spectrogram labels.

# I want to replace the power values in each spectrogram with the associated
# label for that point in time. 

# stacked_windows = simple_tweetyclr.stacked_windows.copy()
# stacked_windows.shape = (stacked_windows.shape[0], 100, 151)

# stacked_windows[:, :, :] = simple_tweetyclr.stacked_labels_for_window[:, :, None]

# stacked_windows.shape = (stacked_windows.shape[0], 100*151) 

# np.save('/Users/AnanyaKapoor/Downloads/stacked_windows_analysis.npy', stacked_windows)
# np.save('/Users/AnanyaKapoor/Downloads/mean_cols_analysis.npy', simple_tweetyclr.mean_colors_per_minispec)

# embed = np.load(f'{simple_tweetyclr.folder_name}/raw_umap_of_specs_with_labels.npy')

# plt.figure()
# plt.scatter(embed[:,0], embed[:,1], c = simple_tweetyclr.mean_colors_per_minispec)
# plt.suptitle("UMAP Decomposition of Spectrograms Using Edit Distance Similarity")
# plt.title(f'Number of Slices: {embed.shape[0]}')
# plt.savefig(f'{simple_tweetyclr.folder_name}/Plots/umap_decomp_of_spec_slice_hamming.png')
# plt.show()


total_dataset = TensorDataset(torch.tensor(simple_tweetyclr.stacked_windows.reshape(simple_tweetyclr.stacked_windows.shape[0], 1, 100, 151)))
batch_size = 36
total_dataloader = DataLoader(total_dataset, batch_size=batch_size , shuffle=False)

# list_of_images = []
# for batch_idx, (data) in enumerate(total_dataloader):
#     data = data[0]

#     for image in data:
#         list_of_images.append(image)

# list_of_images = [tensor.numpy() for tensor in list_of_images]

# embeddable_images = simple_tweetyclr.get_images(list_of_images)

# simple_tweetyclr.plot_UMAP_embedding(embed, simple_tweetyclr.mean_colors_per_minispec, embeddable_images, f'{simple_tweetyclr.folder_name}/Plots/UMAP_of_specs_with_labels.html', saveflag = True)


# In[14]: Let's create a siamese network to predict the hamming distance

# I need to modify this architecture so that I am doing the following
# 1. Batch size is user specified
# 2. Loss is calculated as the pairwise absolute distance between observations in the batch 

# This approach is a bit different than having a batch size of 2 and directly
# predicting the hamming distance. I tried this approach in order to use 
# computational power a bit better 

# Question: is a model that is trained to match the batch pairwise hamming 
# distance conducive to good phrase representations? 

import torch
import torch.nn as nn
import torch.optim as optim

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 3,1,padding=1)
        self.conv2 = nn.Conv2d(8, 8, 3,2,padding=1)
        self.conv3 = nn.Conv2d(8, 16,3,1,padding=1)
        self.conv4 = nn.Conv2d(16,16,3,2,padding=1)
        self.conv5 = nn.Conv2d(16,24,3,1,padding=1)
        self.conv6 = nn.Conv2d(24,24,3,2,padding=1)
        self.conv7 = nn.Conv2d(24,32,3,1,padding=1)
        self.conv8 = nn.Conv2d(32,24,3,2,padding=1)
        self.conv9 = nn.Conv2d(24,24,3,1,padding=1)
        self.conv10 = nn.Conv2d(24,16,3,2,padding=1)

        self.bn1 = nn.BatchNorm2d(8)
        self.bn2 = nn.BatchNorm2d(8)
        self.bn3 = nn.BatchNorm2d(16)
        self.bn4 = nn.BatchNorm2d(16)
        self.bn5 = nn.BatchNorm2d(24)
        self.bn6 = nn.BatchNorm2d(24)
        self.bn7 = nn.BatchNorm2d(32)
        self.bn8 = nn.BatchNorm2d(24)
        self.bn9 = nn.BatchNorm2d(24)
        self.bn10 = nn.BatchNorm2d(16)

        # self.relu = nn.ReLU()
        self.relu = nn.LeakyReLU(0.01)
        
        self.fc = nn.Linear(320, 1)

        self._to_linear = 1
        
        
        # Initialize convolutional layers with He initialization
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.zeros_(m.bias)

    def forward(self, x):

        # x = F.relu((self.conv1(x)))
        # x = F.relu((self.conv2(x)))
        # x = F.relu((self.conv3(x)))
        # x = F.relu((self.conv4(x)))
        # x = F.relu((self.conv5(x)))
        # x = F.relu((self.conv6(x)))
        # x = F.relu((self.conv7(x)))
        # x = F.relu((self.conv8(x)))
        # x = F.relu((self.conv9(x)))
        # x = F.relu((self.conv10(x)))

    
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.relu(self.bn5(self.conv5(x)))
        x = self.relu(self.bn6(self.conv6(x)))
        x = self.relu(self.bn7(self.conv7(x)))
        x = self.relu(self.bn8(self.conv8(x)))
        x = self.relu(self.bn9(self.conv9(x)))
        x = self.relu(self.bn10(self.conv10(x)))

        x_flattened = x.view(-1, 320)

        x = self.relu(self.fc(x_flattened))
        
        pairwise_differences = [torch.abs(obs2 - obs1) for obs1, obs2 in itertools.combinations(x, 2)]
        pairwise_differences = torch.cat(pairwise_differences, dim = 0)

        hamming_like_distance = torch.sum(pairwise_differences)

        # hamming_like_distance = torch.sum(torch.abs(x[0,:] - x[1,:]))
        
        return hamming_like_distance


simple_tweetyclr.shuffling(295)
# torch.manual_seed(295)
shuffled_indices = simple_tweetyclr.shuffled_indices

# stacked_windows = simple_tweetyclr.stacked_windows[shuffled_indices,:]
stacked_windows = simple_tweetyclr.stacked_windows.copy()
# labels = simple_tweetyclr.stacked_labels_for_window[shuffled_indices, :]
labels = simple_tweetyclr.stacked_labels_for_window.copy()

from torch.utils.data import random_split

dataset = torch.tensor(stacked_windows.reshape(stacked_windows.shape[0], 1, 100, 151))

dataset = TensorDataset(dataset, torch.tensor(labels))

total_dataloader = DataLoader(dataset, batch_size=batch_size , shuffle=False)


# Define the split sizes
train_perc = 0.8
train_size = int(train_perc * len(dataset))  # 80% for training
test_size = len(dataset) - train_size  # 20% for testing

# Split the dataset
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

from torch.utils.data import DataLoader

shuffle_status = False


train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_status)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle_status)

model = Encoder()
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()  # MSE Loss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
accumulation_steps = 2
num_epochs = 500

predictions = []

epoch_loss_train = []
epoch_loss_test = []
for epoch in range(num_epochs):
    model.train()  # Set model to training mode
    batch_loss_train = 0
    batch_loss_test = 0
    # Initialize gradient accumulation
    optimizer.zero_grad()
    
    for i, (inputs, labels) in enumerate(train_loader):
        # Transfer inputs and labels to the GPU
        inputs = inputs.to(device, dtype=torch.float32)
        labels = labels.to(device, dtype=torch.float32)
        
        pred_hamming = model(inputs)
        predictions.append(pred_hamming.item())
        
        actual_hamming = [(obs2 != obs1).sum().unsqueeze(0) for obs1, obs2 in itertools.combinations(labels, 2)]
        actual_hamming = torch.cat(actual_hamming, dim = 0)
        actual_hamming = torch.sum(actual_hamming).to(torch.float32)
        
        loss = criterion(pred_hamming, actual_hamming)
        
        # Scale the loss (if needed) and perform backward pass
        # Note: You scale the loss if the batch size is effectively being increased by accumulation
        scaled_loss = loss / accumulation_steps
        scaled_loss.backward()

        batch_loss_train += loss.item()

        # Perform an optimization step after accumulating the specified number of gradients
        if (i + 1) % accumulation_steps == 0 or i + 1 == len(train_loader):
            optimizer.step()
            optimizer.zero_grad()  # Reset gradients after optimizer step
            
        batch_loss_train += loss.item()
        
        # break
        
        
    # model.eval()
        
    # for i, (inputs, labels) in enumerate(test_loader):
    #     inputs = inputs.to(device, dtype=torch.float32)
    #     labels = labels.to(device, dtype=torch.float32)
        
    #     pred_hamming = model(inputs)
        
    #     actual_hamming = [(obs2 != obs1).sum().unsqueeze(0) for obs1, obs2 in itertools.combinations(labels, 2)]
    #     actual_hamming = torch.cat(actual_hamming, dim = 0)/100
    #     actual_hamming = torch.sum(actual_hamming).to(torch.float32)
        
    #     # actual_hamming = (labels[0,:] != labels[1,:]).sum().to(torch.float32)
    #     # actual_hamming = actual_hamming.view(1, 1)
    #     loss = criterion(pred_hamming, actual_hamming)
        
    #     # Scales the loss, and calls backward() to accumulate gradients
    #     loss = loss / accumulation_steps
        
    #     batch_loss_test += loss.item()
        
    #     break
        
    # Logging the loss averaged over an epoch
    # epoch_loss_train.append(batch_loss_train)
    # epoch_loss_test.append(batch_loss_test)
    epoch_loss_train.append(batch_loss_train/len(train_loader))
    # epoch_loss_test.append(batch_loss_test/len(test_loader))
    # epoch_loss.append(batch_loss / len(train_loader))

    # print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_loss_train[-1]:.4f}, Test Loss: {epoch_loss_test[-1]:.4f}')
    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_loss_train[-1]:.4f}')


epoch_loss_train_arr = np.array(epoch_loss_train)
# epoch_loss_test_arr = np.array(epoch_loss_test)

plt.figure()
plt.plot(np.log(epoch_loss_train_arr + 1e-5), label = 'Train Loss')
# plt.plot(np.log(epoch_loss_test_arr + 1e-5), label = 'Validation Loss')
plt.axhline(np.log(1e-5), color = 'red', linestyle = '--', label = 'Lowest Possible Loss')
plt.legend()
plt.title("Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("log(loss + 1e-5)")
plt.show()

# Let's save the parameters from this experiment (this will go in the utils
# file later)

# Here are the parameters I want to save:
# 1. Data parameters: 
    # a. Number of spectrograms for analysis
    # b. Stride
    # c. Window size
    # d. Total number of spectrogram slices
    # e. Whether any standardization was applied

# 2. Model parameters: 
    # a. Optimizer type and Learning rate
    # b. Batch size 
    # c. Number of epochs
    # d. Random seed
    # e. Accumulation? 
    # f. Train/Test split? If so, what is the proportion?
    # g. Model architecture & any regularlization. 
    # h. Criterion
    
# data_params = {
#     "Data_Directory": bird_dir,
#     "Window_Size": simple_tweetyclr.window_size, 
#     "Stride_Size": simple_tweetyclr.stride, 
#     "Num_Spectrograms": simple_tweetyclr.num_spec, 
#     "Total_Slices": simple_tweetyclr.stacked_windows.shape[0], 
#     "Frequencies_of_Interest": masking_freq_tuple, 
#     "Data_Standardization": "None"
#     }

model_arch = str(model)
forward_method = inspect.getsource(model.forward)

# Splitting the string into an array of lines
model_arch_lines = model_arch.split('\n')
forward_method_lines = forward_method.split('\n')


experiment_params = {
    "Data_Directory": bird_dir,
    "Window_Size": simple_tweetyclr.window_size, 
    "Stride_Size": simple_tweetyclr.stride, 
    "Num_Spectrograms": simple_tweetyclr.num_spec, 
    "Total_Slices": simple_tweetyclr.stacked_windows.shape[0], 
    "Frequencies_of_Interest": masking_freq_tuple, 
    "Data_Standardization": "None",
    "Optimizer": str(optimizer), 
    "Batch_Size": batch_size, 
    "Num_Epochs": num_epochs, 
    "Torch_Random_Seed": 295, 
    "Accumulation_Size": 1, 
    "Train_Proportion": train_perc,
    "Criterion": str(criterion), 
    "Model_Architecture": model_arch_lines, 
    "Forward_Method": forward_method_lines, 
    "Dataloader_Shuffle": True
    }

import json

with open(f'{simple_tweetyclr.folder_name}/experiment_params.json', 'w') as file:
    json.dump(experiment_params, file, indent=4)

# model_rep = []

# model.eval()
# with torch.no_grad():
#     for i, (inputs, labels) in enumerate(total_dataloader):
#         inputs = inputs.to(device, dtype=torch.float32)
#         labels = labels.to(device, dtype=torch.float32)
    
#         x = F.relu(model.bn1(model.conv1(inputs)))
#         x = F.relu(model.bn2(model.conv2(x)))
#         x = F.relu(model.bn3(model.conv3(x)))
#         x = F.relu(model.bn4(model.conv4(x)))
#         x = F.relu(model.bn5(model.conv5(x)))
#         x = F.relu(model.bn6(model.conv6(x)))
#         x = F.relu(model.bn7(model.conv7(x)))
#         x = F.relu(model.bn8(model.conv8(x)))
#         x = F.relu(model.bn9(model.conv9(x)))
#         x = F.relu(model.bn10(model.conv10(x)))

#         x_flattened = x.view(-1, 320)
#         model_rep.append(x_flattened.clone().detach().numpy())
        
# model_rep1 = np.concatenate((model_rep))    
        
# Write the experiment information to a txt file 







