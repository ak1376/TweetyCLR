#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 11:11:09 2023

UMAP analysis of canary song spectrogram slices. Instead of doing a UMAP analysis on the spectrogram slices, I am doing a UMAP on the slice labels.

@author: AnanyaKapoor
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
analysis_path = f'{filepath}/Dropbox (University of Oregon)/Kapoor_Ananya/01_Projects/01_b_Canary_SSL/TweetyCLR_End_to_End/'
# analysis_path = '/home/akapoor/Dropbox (University of Oregon)/Kapoor_Ananya/01_Projects/01_b_Canary_SSL/TweetyCLR_Repo/'

# # Parameters we set
num_spec = 1
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

# In[2]: UMAP on spectrogram labels.

# I want to replace the power values in each spectrogram with the associated
# label for that point in time. 

stacked_windows = simple_tweetyclr.stacked_windows.copy()
stacked_windows.shape = (stacked_windows.shape[0], 100, 151)

stacked_windows[:, :, :] = simple_tweetyclr.stacked_labels_for_window[:, :, None]

stacked_windows.shape = (stacked_windows.shape[0], 100*151) 

# Set up a base dataloader (which we won't directly use for modeling). Also define the batch size of interest
total_dataset = TensorDataset(torch.tensor(simple_tweetyclr.stacked_windows.reshape(simple_tweetyclr.stacked_windows.shape[0], 1, 100, 151)))
batch_size = 1
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

        self.relu = nn.ReLU(inplace = True)
        self.sigmoid = nn.Sigmoid()
        
        self.fc = nn.Sequential(
            nn.Linear(640, 256),
            nn.ReLU(inplace=True), 
            nn.Linear(256, 1)
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

    def forward(self, input1, input2):
        
        # Pass the two spectrogram slices through a convolutional frontend to get a representation for each slice
        
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        
        # concatenate both images' features
        output = torch.cat((output1, output2), 1)

        # pass the concatenation to the linear layers
        output = self.fc(output)

        # pass the out of the linear layers to sigmoid layer
        output = self.relu(output)
        
        return output
    
class APP_MATCHER(Dataset):
    def __init__(self, dataset):
        super(APP_MATCHER, self).__init__()
        
        # Extracting all features and targets
        all_features, all_targets = zip(*[dataset[i] for i in range(len(dataset))])
        
        # Converting lists of tensors to a single tensor
        all_features = torch.stack(all_features)
        all_targets = torch.stack(all_targets)

        
        self.dataset = all_features
        self.targets = all_targets
        self.data = all_features.clone()
        
        self.total_indices = np.arange(len(self.dataset))
        self.group_examples()
        self.index_to_corresponding_indices = self._create_index_map()
        self.different_class_indices = self._create_different_labels_map()

    def _create_index_map(self):
        index_map = {}
        for key, indices in self.final_dict.items():
            for index in indices:
                index_map[index] = indices
        return index_map
    
    def _create_different_labels_map(self):
        
        different_labels_map = {}
    
        all_indices = set(range(len(self.dataset)))
        for key, indices in self.final_dict.items():
            different_label_indices = list(all_indices - set(indices))
            for index in indices:
                different_labels_map[index] = different_label_indices
    
        return different_labels_map
        
    def group_examples(self):
        """
        To ease the accessibility of data based on the class, we will use `group_examples` to group 
        examples based on class. 
        
        Class definition here is the number of unique combinations of window labels. 
        
        This is a toy example, but for 72 spectrogram slices, there are 7 unique label combinations:
            
            1. [0, 29]
            2. [0, 1, 29]
            3. [0, 1]
            4. [0, 1, 2]
            5. [0, 2]
            6. [0, 2, 11]
            7. [0, 11]
            
        If I pick two spectrogram slices with the same unique label combination then I would want to represent them similarly. 

        """

        # get the targets from dataset
        np_arr = np.array(self.targets.clone())
        
        # group examples based on class
        self.grouped_examples = {}
        
        # For now I am going to implement on toy dataset
        unique_labels_per_row = [np.unique(row) for row in np.array(self.targets)]
        
        # Creating a dictionary to store indices for each unique array
        index_dict = defaultdict(list)
        
        for i, arr in enumerate(unique_labels_per_row):
            index_dict[tuple(arr)].append(i)
        
        # Converting tuples back to arrays in the dictionary keys
        final_dict = {key: value for key, value in index_dict.items()}
        
        lengths = [len(value) for value in final_dict.values()]
        
        self.final_dict = final_dict
        
        
    def __len__(self):
        return self.data.shape[0]
    
    
    def __getitem__(self, index):
        ''' 
        For the spectrogram slice corresponding to "index", I want to randomly
        choose another spectrogram slice. If that randomly chosen spectrogram
        slice has the same collection of labels then the pseudo-label will be
        1. Otherwise, the pseudo-label will be 0
        '''
        
        # Choose a random row of the targets array and extract out the labels
        # in that row
        
        random_index = random.randint(0, len(self.dataset) - 1)

        
        # Let's extract a spectrogram slice with the same collection of labels
        # Finding the key corresponding to the given index
        corresponding_indices = self.index_to_corresponding_indices[index]

        index_1 = random.choice(corresponding_indices) # This is our anchor
        
        image_1 = self.dataset[index_1,:]
        label_1 = self.targets[index_1, :]
        
        # same class
        if index % 2 == 0:
            # Optimized logic for selecting index_2
            index_2 = index_1
            if len(corresponding_indices) > 1:
                index_2 = random.choice([i for i in corresponding_indices if i != index_1])

            image_2 = self.dataset[index_2,:]
            
            
        else:
            # Optimized different label selection (assuming precomputed mapping)
            indices_of_different_labels = self.different_class_indices[index]
            index_2 = random.choice(indices_of_different_labels)
            image_2 = self.dataset[index_2,:]

        label_2 = self.targets[index_2, :]
        target = (label_1 != label_2).sum().unsqueeze(0).to(torch.float32).sum()
        indices_list = torch.tensor([index_1, index_2])
         
        return image_1, image_2, target, indices_list
    
# Convert the dataset and targets into a torch dataset from which we can easily divide into training and testing

stacked_windows = simple_tweetyclr.stacked_windows.copy()
stacked_labels_for_window = simple_tweetyclr.stacked_labels_for_window.copy()

dataset = TensorDataset(torch.tensor(simple_tweetyclr.stacked_windows), torch.tensor(simple_tweetyclr.stacked_labels_for_window))
# dataset = TensorDataset(torch.tensor(stacked_windows), torch.tensor(stacked_labels_for_window))

# Split the dataset into a training and testing dataset
# Define the split sizes -- what is the train test split ? 
train_perc = 0.8 #
train_size = int(train_perc * len(dataset))  # (100*train_perc)% for training
test_size = len(dataset) - train_size  # 100 - (100*train_perc)% for testing

from torch.utils.data import random_split

train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_dataset = APP_MATCHER(train_dataset)
test_dataset = APP_MATCHER(test_dataset)

shuffle_status = True

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle = shuffle_status)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = batch_size, shuffle = shuffle_status)

model = Encoder()
# Check if multiple GPUs are available
if torch.cuda.device_count() > 1:
    print(f"Number of GPUs available: {torch.cuda.device_count()}")
    model = nn.DataParallel(model)

# Move the model to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
    
        
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss() 


num_epochs = 100
patience = 10  # Number of epochs to wait for improvement before stopping
min_delta = 0.001  # Minimum change to qualify as an improvement

best_val_loss = float('inf')
epochs_no_improve = 0
early_stop = False
tic = time.time()
training_epoch_loss = []
validation_epoch_loss = []
for epoch in np.arange(num_epochs):
    model.train()
    training_loss = 0
    for batch_idx, (images_1, images_2, targets, indices) in enumerate(train_loader):
        images_1, images_2, targets = images_1.to(device, dtype = torch.float32), images_2.to(device, dtype = torch.float32), targets.to(device, dtype = torch.float32)
        images_1 = images_1.reshape(images_1.shape[0], 100, 151).unsqueeze(1)
        images_2 = images_2.reshape(images_2.shape[0], 100, 151).unsqueeze(1)
        
        optimizer.zero_grad()
        outputs = model(images_1, images_2).squeeze()
        loss = criterion(outputs, targets.squeeze())
        training_loss+=loss.item()
        loss.backward()
        optimizer.step()
        
    model.eval()
    with torch.no_grad():
        validation_loss = 0
        for batch_idx, (images_1, images_2, targets, indices) in enumerate(test_loader):
            images_1, images_2, targets = images_1.to(device, dtype = torch.float32), images_2.to(device, dtype = torch.float32), targets.to(device, dtype = torch.float32)
            images_1 = images_1.reshape(images_1.shape[0], 100, 151).unsqueeze(1)
            images_2 = images_2.reshape(images_2.shape[0], 100, 151).unsqueeze(1)
            outputs = model(images_1, images_2).squeeze()
            loss = criterion(outputs, targets.squeeze())
            validation_loss+=loss.item()
        
    training_epoch_loss.append(training_loss / len(train_loader))
    validation_epoch_loss.append(validation_loss / len(test_loader))
    
    # Check for improvement
    if validation_epoch_loss[-1] < best_val_loss - min_delta:
        best_val_loss = validation_epoch_loss[-1]
        epochs_no_improve = 0
        torch.save(model.state_dict(), f'{folder_name}/model_state_dict_epoch_{epoch}.pth')

    else:
        epochs_no_improve += 1
    
    print(f'Epoch {epoch}, Training Loss: {training_epoch_loss[-1]}, Validation Loss {validation_epoch_loss[-1]}')
    
plt.figure()
plt.plot(training_epoch_loss, label = 'Training Loss')
plt.plot(validation_epoch_loss, label = 'Validation Loss')
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Supervised Training")
plt.savefig(f'{folder_name}/loss_curve.png')
plt.show()

model_form = model.module


model_arch = str(model_form)
forward_method = inspect.getsource(model_form.forward)
forward_once_method = inspect.getsource(model_form.forward_once)

# Splitting the string into an array of lines
model_arch_lines = model_arch.split('\n')
forward_method_lines = forward_method.split('\n')
forward_once_method_lines = forward_once_method.split('\n')

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
    "Forward_Once_Method": forward_once_method_lines,
    "Dataloader_Shuffle": shuffle_status
    }

import json

with open(f'{simple_tweetyclr.folder_name}/experiment_params.json', 'w') as file:
    json.dump(experiment_params, file, indent=4)

toc = time.time()

print('========================')
print(f'Total Time: {toc - tic}')       
        
# Now I want to see the model representation

model_rep = []
total_dataset = TensorDataset(torch.tensor(simple_tweetyclr.stacked_windows.reshape(simple_tweetyclr.stacked_windows.shape[0], 1, 100, 151)), torch.tensor(simple_tweetyclr.stacked_labels_for_window))

# total_dat = APP_MATCHER(total_dataset)
# total_dataloader = torch.utils.data.DataLoader(total_dat, batch_size = batch_size, shuffle = shuffle_status)
model = model.to('cpu')
model.eval()
with torch.no_grad():
    for batch_idx, data in enumerate(total_dataloader):
        data = data[0]
        data = data.to(torch.float32)
        output = model.module.forward_once(data)
        model_rep.append(output.numpy())

model_rep_stacked = np.concatenate((model_rep))

import umap
reducer = umap.UMAP(random_state=295) # For consistency
embed = reducer.fit_transform(model_rep_stacked)
np.save(f'{folder_name}/embedding.npy', embed)


plt.figure()
plt.scatter(embed[:,0], embed[:,1], c = simple_tweetyclr.mean_colors_per_minispec)
plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
plt.title("UMAP of the Representation Layer")
plt.show()
plt.savefig(f'{folder_name}/UMAP_rep_of_model.png')

# Bokeh Plot
list_of_images = []
for batch_idx, (data) in enumerate(total_dataloader):
    data = data[0]
    
    for image in data:
        list_of_images.append(image)
        
list_of_images = [tensor.numpy() for tensor in list_of_images]

embeddable_images = simple_tweetyclr.get_images(list_of_images)

simple_tweetyclr.plot_UMAP_embedding(embed, simple_tweetyclr.mean_colors_per_minispec,embeddable_images, f'{simple_tweetyclr.folder_name}/Plots/UMAP_Analysis.html', saveflag = True)

