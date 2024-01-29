#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 21:10:36 2024

This model will be a simpler version of UMAP + TweetyCLR. Here are the following modifications: 
    1. New model architecture 
    2. No temporal augmentation. Positive sample will be the spec slice that is closest to the specific hard index
    3. One easy negative sample only
    4. No train/test split for now

@author: akapoor
"""

import numpy as np
import torch
import sys
filepath = '/home/akapoor'
import os
# os.chdir('/Users/AnanyaKapoor/Downloads/TweetyCLR')
os.chdir('/home/akapoor/Dropbox (University of Oregon)/Kapoor_Ananya/01_Projects/01_b_Canary_SSL/TweetyCLR_End_to_End')
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
from torch.utils.data import Dataset
from collections import defaultdict
import time


# Set random seeds for reproducibility 
torch.manual_seed(295)
np.random.seed(295)
random.seed(295)

# Matplotlib plotting settings
plt.rcParams.update({'font.size': 20})
plt.rcParams['figure.figsize'] = [15, 15]  # width and height should be in inches

# Specify the necessary directories 
bird_dir = f'{filepath}/Dropbox (University of Oregon)/Kapoor_Ananya/01_Projects/01_a_Rotations/Gardner_Lab/Canary_Data/llb16_data_matrices/'
audio_files = bird_dir+'llb3_songs'
directory = bird_dir+ 'Python_Files'

# Identify the upstream location where the results will be saved. 
analysis_path = f'{filepath}/Dropbox (University of Oregon)/Kapoor_Ananya/01_Projects/01_b_Canary_SSL/TweetyCLR_Repo/'

# # Parameters we set
num_spec = 10
window_size = 100
stride = 10

# Define the folder name

# I want to have a setting where the user is asked whether they want to log an
# experiment. The user should also provide a brief text description of what the
# experiment is testing (like a Readme file)

log_experiment = True
if log_experiment == True:
    user_input = input("Please enter the experiment name: ")
    folder_name = f'{analysis_path}Supervised_Task/Num_Spectrograms_{num_spec}_Window_Size_{window_size}_Stride_{stride}/{user_input}'
    
else:
    folder_name = f'{analysis_path}Supervised_Task/Num_Spectrograms_{num_spec}_Window_Size_{window_size}_Stride_{stride}'


# Organize the files for analysis 
files = os.listdir(directory)
all_songs_data = [f'{directory}/{element}' for element in files if '.npz' in element] # Get the file paths of each numpy file from Yarden's data
all_songs_data.sort()

# Identity any low and high pass filtering 
masking_freq_tuple = (500, 7000)

# Dimensions of the spec slices for analysis 
spec_dim_tuple = (window_size, 151)

# Ground truth label coloring
with open(f'{filepath}/Dropbox (University of Oregon)/Kapoor_Ananya/01_Projects/01_b_Canary_SSL/TweetyCLR_End_to_End/Supervised_Task/category_colors.pkl', 'rb') as file:
    category_colors = pickle.load(file)

# In[1]: Creating Dataset

# Object that has a bunch of helper functions and does a bunch of useful things 
simple_tweetyclr_experiment_1 = Tweetyclr(num_spec, window_size, stride, folder_name, all_songs_data, masking_freq_tuple, spec_dim_tuple, category_colors)

simple_tweetyclr = simple_tweetyclr_experiment_1

# Finds the sliding windows
simple_tweetyclr.first_time_analysis()

# Documentation code
if log_experiment == True: 
    exp_descp = input('Please give a couple of sentences describing what the experiment is testing: ')
    # Save the input to a text file
    with open(f'{folder_name}/experiment_readme.txt', 'w') as file:
        file.write(exp_descp)


stacked_windows = simple_tweetyclr.stacked_windows.copy()
stacked_windows.shape = (stacked_windows.shape[0], 100, 151)

stacked_windows[:, :, :] = simple_tweetyclr.stacked_labels_for_window[:, :, None]

stacked_windows.shape = (stacked_windows.shape[0], 100*151) 

# Set up a base dataloader (which we won't directly use for modeling). Also define the batch size of interest
total_dataset = TensorDataset(torch.tensor(simple_tweetyclr.stacked_windows.reshape(simple_tweetyclr.stacked_windows.shape[0], 1, 100, 151)))
batch_size = 128
total_dataloader = DataLoader(total_dataset, batch_size=batch_size , shuffle=False)

import torch
import torch.nn as nn
import torch.optim as optim
# from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau

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
        
        self.ln1 = nn.LayerNorm([8, 100, 151])
        self.ln2 = nn.LayerNorm([8, 50, 76])
        self.ln3 = nn.LayerNorm([16, 50, 76])
        self.ln4 = nn.LayerNorm([16, 25, 38])
        self.ln5 = nn.LayerNorm([24, 25, 38])
        self.ln6 = nn.LayerNorm([24, 13, 19])
        self.ln7 = nn.LayerNorm([32, 13, 19])
        self.ln8 = nn.LayerNorm([24, 7, 10])
        self.ln9 = nn.LayerNorm([24, 7, 10])
        self.ln10 = nn.LayerNorm([16, 4, 5])

        self.relu = nn.ReLU(inplace = False)
        self.sigmoid = nn.Sigmoid()
        
        # self.fc = nn.Sequential(
        #     nn.Linear(320, 256),
        #     nn.ReLU(inplace=True), 
        #     nn.Linear(256, 1)
        # )  
        self.fc = nn.Sequential(
            nn.Linear(320, 256),
            nn.ReLU(inplace=False), 
        )  
        
        self.dropout = nn.Dropout(p=0.5)
        
        # Initialize convolutional layers with He initialization
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.zeros_(m.bias)

    def forward_once(self, x):

        # No BatchNorm
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

        # BatchNorm 
        # x = self.relu(self.bn1(self.conv1(x)))
        # x = self.relu(self.bn2(self.conv2(x)))
        # x = self.relu(self.bn3(self.conv3(x)))
        # x = self.relu(self.bn4(self.conv4(x)))
        # x = self.relu(self.bn5(self.conv5(x)))
        # x = self.relu(self.bn6(self.conv6(x)))
        # x = self.relu(self.bn7(self.conv7(x)))
        # x = self.relu(self.bn8(self.conv8(x)))
        # x = self.relu(self.bn9(self.conv9(x)))
        # x = self.relu(self.bn10(self.conv10(x)))
        
        # LayerNorm
        # x = (self.relu(self.ln1(self.conv1(x))))
        # x = (self.relu(self.ln2(self.conv2(x))))
        # x = (self.relu(self.ln3(self.conv3(x))))
        # x = (self.relu(self.ln4(self.conv4(x))))
        # x = (self.relu(self.ln5(self.conv5(x))))
        # x = (self.relu(self.ln6(self.conv6(x))))
        # x = (self.relu(self.ln7(self.conv7(x))))
        # x = (self.relu(self.ln8(self.conv8(x))))
        # x = (self.relu(self.ln9(self.conv9(x))))
        # x = (self.relu(self.ln10(self.conv10(x))))
        
        # LayerNorm + Dropout
        x = self.dropout(self.relu(self.ln1(self.conv1(x))))
        x = self.dropout(self.relu(self.ln2(self.conv2(x))))
        x = self.dropout(self.relu(self.ln3(self.conv3(x))))
        x = self.dropout(self.relu(self.ln4(self.conv4(x))))
        x = self.dropout(self.relu(self.ln5(self.conv5(x))))
        x = self.dropout(self.relu(self.ln6(self.conv6(x))))
        x = self.dropout(self.relu(self.ln7(self.conv7(x))))
        x = self.dropout(self.relu(self.ln8(self.conv8(x))))
        x = self.dropout(self.relu(self.ln9(self.conv9(x))))
        x = self.dropout(self.relu(self.ln10(self.conv10(x))))

        x_flattened = x.view(-1, 320)
        
        return x_flattened
    
    def forward(self, anchor_img, positive_img, negative_img):
        
        # Pass the two spectrogram slices through a convolutional frontend to get a representation for each slice
        
        anchor_emb = self.relu(self.fc(self.forward_once(anchor_img)))
        positive_emb = self.relu(self.fc(self.forward_once(positive_img)))
        negative_emb = self.relu(self.fc(self.forward_once(negative_img)))
        
        return anchor_emb, positive_emb, negative_emb


# Need to compute the UMAP embedding
# reducer = umap.UMAP(metric = 'cosine', random_state=295)
reducer = umap.UMAP(metric = 'cosine')

embed = reducer.fit_transform(simple_tweetyclr.stacked_windows)
# Preload the embedding 
# embed = np.load(f'{simple_tweetyclr.folder_name}/embed_80_specs.npy')
simple_tweetyclr.umap_embed_init = embed

plt.figure()
plt.scatter(embed[:,0], embed[:,1], s = 10, c = simple_tweetyclr.mean_colors_per_minispec)
plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
plt.title(f'Total Slices: {embed.shape[0]}')
plt.savefig(f'{simple_tweetyclr.folder_name}/Plots/UMAP_of_all_slices.png')
plt.show()

# DEFINE HARD INDICES THROUGH INTERACTION: USER NEEDS TO ZOOM IN ON ROI 

# If we are including multiple hard regions then I should put them in a 
# dictionary and then select hard and easy negative samples according to which
# key we are in in in the dictionary

hard_indices_dict = {}
hard_region_coordinates = {}

# Get current axes
ax = plt.gca()
# Get current limits
xlim = ax.get_xlim()
ylim = ax.get_ylim()

hard_indices = np.where((embed[:,0]>=xlim[0])&(embed[:,0]<=xlim[1]) & (embed[:,1]>=ylim[0]) & (embed[:,1]<=ylim[1]))[0]


hard_indices_dict[0] = hard_indices
hard_region_coordinates[0] = [xlim, ylim]

# Now let's zoom out on the matplotlib plot

# Zoom out by changing the limits
# You can adjust these values as needed to zoom out to the desired level
ax.set_xlim([min(embed[:,0]) - 1, max(embed[:,0]) + 1])  # Zoom out on x-axis
ax.set_ylim([min(embed[:,1]) - 1, max(embed[:, 1]) + 1])  # Zoom out on y-axis

# Show the updated plot
plt.show()


for i in np.arange(len(hard_indices_dict)):
    hard_ind = hard_indices_dict[i]

    plt.figure(figsize = (10,10))
    plt.scatter(embed[hard_ind,0], embed[hard_ind,1], s = 10, c = simple_tweetyclr.mean_colors_per_minispec[hard_ind,:])
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    # plt.title("UMAP Decomposition of ")
    plt.suptitle(f'UMAP Representation of Hard Region #{i}')
    plt.title(f'Total Slices: {embed[hard_ind,:].shape[0]}')
    plt.savefig(f'{simple_tweetyclr.folder_name}/Plots/UMAP_of_hard_slices_region_{i}.png')
    plt.show()


data_for_analysis = simple_tweetyclr.stacked_windows.copy()
data_for_analysis = data_for_analysis.reshape(data_for_analysis.shape[0], 1, 100, 151)
total_dataset = TensorDataset(torch.tensor(data_for_analysis.reshape(data_for_analysis.shape[0], 1, simple_tweetyclr.time_dim, simple_tweetyclr.freq_dim)))
total_dataloader = DataLoader(total_dataset, batch_size=batch_size , shuffle=False)

list_of_images = []
for batch_idx, (data) in enumerate(total_dataloader):
    data = data[0]
    
    for image in data:
        list_of_images.append(image)
        
list_of_images = [tensor.numpy() for tensor in list_of_images]

embeddable_images = simple_tweetyclr.get_images(list_of_images)

simple_tweetyclr.plot_UMAP_embedding(embed, simple_tweetyclr.mean_colors_per_minispec,embeddable_images, f'{simple_tweetyclr.folder_name}/Plots/UMAP_of_all_slices.html', saveflag = True)

hard_dataset = data_for_analysis[hard_indices,:].reshape(len(hard_indices), 1, 100, 151)

dataset = TensorDataset(torch.tensor(data_for_analysis), torch.tensor(np.arange(data_for_analysis.shape[0])))
hard_dataset = TensorDataset(torch.tensor(hard_dataset), torch.tensor(hard_indices))


# In[2]: I want to define a class where I select positive and negative samples. Here is the criterion for each
# Positive sample: A spectrogram slice that is closest in proximity to the anchor datapoint
# Negative sample: A spectrogram slice that is outside the hard_indices region

class APP_MATCHER(Dataset):
    def __init__(self, dataset, hard_dataset, umap_embedding):
        ''' 
        The APP_MATCHER object should take in the entire dataset and the dataset of hard indices only. 
        The datasets should be Pytorch datasets. THe first component of each 
        dataset will be the actual flattened spectrogram slices. The second 
        component will be the index for each slice from the entire dataset. 
        This will be useful when selecting positive and negative samples.
        '''
        super(APP_MATCHER, self).__init__()
        
        # Extracting all slices and indices
        all_features, all_indices = zip(*[dataset[i] for i in range(len(dataset))])
        
        hard_features, hard_indices = zip(*[hard_dataset[i] for i in range(len(hard_dataset))])
        
        # Converting lists of tensors to a single tensor
        all_features, all_indices = torch.stack(all_features), torch.stack(all_indices)
        hard_features, hard_indices = torch.stack(hard_features), torch.stack(hard_indices)
        
        
        self.dataset = dataset
        self.hard_dataset = hard_dataset
        
        self.all_features = all_features
        self.all_indices = all_indices
        
        self.hard_features = hard_features
        self.hard_indices = hard_indices
        self.umap_embedding = torch.tensor(umap_embedding)
        
        # Find the set of "easy" negatives
        mask = ~torch.isin(self.all_indices, self.hard_indices)
        all_possible_negatives = self.all_indices[mask]
        
        self.all_possible_negatives = all_possible_negatives
        
        # Find the positive images
        positive_dict = self.select_positive_image()
        self.positive_dict = positive_dict
        
    def select_positive_image(self):
        # Compute pairwise Euclidean distances
        distances = torch.cdist(self.umap_embedding[self.hard_indices,:], self.umap_embedding[self.hard_indices,:], p=2)
        
        # Dictionary to store the top 5 closest points for each data point
        # TODO: I should also create a dictionary that goes from 0, ..., 470 and stores the hard_index. That way I can use index in the__getitem__ function to extract the top_5 spectrogram slices.
        top_5_dict = {}
        
        # Iterate over each data point
        for i in range(self.umap_embedding[hard_indices,:].size(0)):
            # Exclude the distance to the point itself by setting it to a high value
            distances[i, i] = float('inf')
        
            # Get indices of the top 5 closest points
            _, closest_indices = torch.topk(distances[i], 5, largest=False)
        
            # Store the corresponding points in the dictionary
            top_5_dict[self.hard_indices[i].item()] = closest_indices
            
        return top_5_dict
    
    def __len__(self):
        return self.hard_indices.shape[0]
    
    def __getitem__(self, index):
        
        ''' 
        The positive sample for each anchor image in the batch will be the spectrogram slice that is closest to the anchor spec slice
        The negative sample for each anchor image in the batch will be a randomly chosen spectrogram slice outside the hard region
        '''
        # TODO: Revise this function so that I don't get KeyErrors anymore
        
        # The positive spectrogram slice will be the spectrogram slice that is closest in UMAP space to the anchor slice.
        
        actual_index = self.hard_indices[index].item()
        anchor_img = self.hard_features[actual_index,:, :, :].unsqueeze(1)
        
        # =============================================================================
        #         Positive Sample
        # =============================================================================
        
        top_5_images = self.positive_dict[actual_index]
        
        # Generate a random index
        random_index = torch.randint(0, top_5_images.size(0), (1,)).item()
        
        # Select the element at the random index
        positive_index = top_5_images[random_index].item()
        positive_img = self.hard_features[positive_index, :, :, :].unsqueeze(1)
                
        # =============================================================================
        #         Negative Sample
        # =============================================================================
        
        random_index = torch.randint(0, self.all_possible_negatives.size(0), (1,)).item()
        
        negative_index = self.all_possible_negatives[random_index].item()
        
        negative_img = self.all_features[negative_index, :, :, :]
        
        return anchor_img, positive_img, negative_img
        
train_dataset = APP_MATCHER(dataset, hard_dataset, embed)   
    
shuffle_status = True

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle = shuffle_status)

a = next(iter(train_loader))

model = Encoder()
# Check if multiple GPUs are available
if torch.cuda.device_count() > 1:
    print(f"Number of GPUs available: {torch.cuda.device_count()}")
    model = nn.DataParallel(model)

# Move the model to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device).to(torch.float32)







        
        

