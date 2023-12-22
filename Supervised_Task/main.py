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

plt.rcParams.update({'font.size': 20})
plt.rcParams['figure.figsize'] = [15, 15]  # width and height should be in inches, e.g., [10, 6]

bird_dir = f'{filepath}/Dropbox (University of Oregon)/Kapoor_Ananya/01_Projects/01_a_Rotations/Gardner_Lab/Canary_Data/llb16_data_matrices/'
audio_files = bird_dir+'llb3_songs'
directory = bird_dir+ 'Python_Files'

analysis_path = f'{filepath}/Dropbox (University of Oregon)/Kapoor_Ananya/01_Projects/01_b_Canary_SSL/TweetyCLR_End_to_End/'
# analysis_path = '/home/akapoor/Dropbox (University of Oregon)/Kapoor_Ananya/01_Projects/01_b_Canary_SSL/TweetyCLR_Repo/'

# # Parameters we set
num_spec = 80
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

# In[23]:


simple_tweetyclr_experiment_1 = Tweetyclr(num_spec, window_size, stride, folder_name, all_songs_data, masking_freq_tuple, spec_dim_tuple, category_colors)
# simple_tweetyclr_experiment_1.temperature_value = temp_value
simple_tweetyclr = simple_tweetyclr_experiment_1
simple_tweetyclr.first_time_analysis()


embed = np.load(f'{simple_tweetyclr.folder_name}/embed_of_labels_hamming.npy')

# plt.figure()
# plt.scatter(embed[:,0], embed[:,1], c = simple_tweetyclr.mean_colors_per_minispec)
# plt.suptitle("UMAP Decomposition of Spectrogram Slice Syllable Labels")
# plt.title(f'Number of Slices: {embed.shape[0]}')
# # plt.savefig(f'{simple_tweetyclr.folder_name}/Plots/umap_decomp_of_spec_slice_labels.png')
# plt.show()


total_dataset = TensorDataset(torch.tensor(simple_tweetyclr.stacked_windows.reshape(simple_tweetyclr.stacked_windows.shape[0], 1, 100, 151)))
batch_size = 64
total_dataloader = DataLoader(total_dataset, batch_size=batch_size , shuffle=False)


# Let's create the images necessary for the Bokeh visualization

# In[13]:


# list_of_images = []
# for batch_idx, (data) in enumerate(total_dataloader):
#     data = data[0]

#     for image in data:
#         list_of_images.append(image)

# list_of_images = [tensor.numpy() for tensor in list_of_images]

# embeddable_images = simple_tweetyclr.get_images(list_of_images)

# simple_tweetyclr.plot_UMAP_embedding(embed, simple_tweetyclr.mean_colors_per_minispec, embeddable_images, f'{simple_tweetyclr.folder_name}/Plots/UMAP_of_window_labels_hamming.html', saveflag = True)

# In[14]: Let's create a siamese network to predict the hamming distance

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

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout2d(
        )
        self.fc = nn.Linear(320, 1)

        self._to_linear = 1
        self.final_layer = nn.Linear(1 * 2, 1) 
        

    def forward_method(self, x):

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

    
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = F.relu(self.bn7(self.conv7(x)))
        x = F.relu(self.bn8(self.conv8(x)))
        x = F.relu(self.bn9(self.conv9(x)))
        x = F.relu(self.bn10(self.conv10(x)))

        # x = x.view(-1, 320)
        x_flattened = x.view(1, -1)
        # x = x.permute(0, x.shape[1]*x.shape[2]*x.shape[3])
        x = F.relu(self.fc(x_flattened))
        
        # hamming_like_distance = torch.sum(torch.abs(x[0,:] - x[1,:]))


        # return hamming_like_distance
        
        return x
    
    def forward(self, input1, input2):
        zi = self.forward_method(input1)
        zj = self.forward_method(input2)
        
        dist = zi - zj
        dist = torch.abs(dist)  # Element-wise absolute value
        hamming_dist = dist
        
        return hamming_dist
        

simple_tweetyclr.shuffling(295)
# torch.manual_seed(295)
shuffled_indices = simple_tweetyclr.shuffled_indices

from torch.utils.data import random_split

dataset = torch.tensor(simple_tweetyclr.stacked_windows.reshape(simple_tweetyclr.stacked_windows.shape[0], 1, 100, 151))

dataset = TensorDataset(dataset, torch.tensor(simple_tweetyclr.stacked_labels_for_window))


# Define the split sizes
train_size = int(0.8 * len(dataset))  # 80% for training
test_size = len(dataset) - train_size  # 20% for testing

# Split the dataset
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

from torch.utils.data import DataLoader

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)

model = Encoder()
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()  # MSE Loss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
accumulation_steps = 32
num_epochs = 500

epoch_loss = []
for epoch in range(num_epochs):
    model.train()  # Set model to training mode
    batch_loss = 0
    for i, (inputs, labels) in enumerate(train_loader):
        # Transfer inputs and labels to the GPU
        inputs = inputs.to(device, dtype=torch.float32)
        labels = labels.to(device, dtype=torch.float32)
        
        input1, input2 = inputs[0,:,:,:], inputs[1,:,:,:]
        input1 = input1.unsqueeze(0)
        input2 = input2.unsqueeze(0)
        
        pred_hamming = model(input1, input2)
        
        actual_hamming = (labels[0,:] != labels[1,:]).sum().to(torch.float32)
        actual_hamming = actual_hamming.view(1, 1)
        loss = criterion(pred_hamming, actual_hamming)

        # Scales the loss, and calls backward() to accumulate gradients
        loss = loss / accumulation_steps
        loss.backward()

        if (i + 1) % accumulation_steps == 0:
            # Perform optimization step after accumulating gradients
            optimizer.step()
            optimizer.zero_grad()

        batch_loss += loss.item()

    # Logging the loss averaged over an epoch
    epoch_loss.append(batch_loss / len(train_loader))

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss[-1]}')
    
    
plt.figure()
plt.plot(epoch_loss)
plt.show()










