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

# I want to use the Edit distance for UMAP decomposition. Need to write a custom function

import numpy as np
import numba
@numba.njit()
def custom_distance(x, y):
    """
    Calculate the custom distance metric equivalent to (labels[0,:] != labels[1,:]).sum() in PyTorch.

    Parameters:
    x, y (numpy arrays): Two label vectors to compare.

    Returns:
    float: The calculated distance.
    """
    # Ensure the inputs are NumPy arrays
    x = np.asarray(x)
    y = np.asarray(y)

    # Calculate the number of different elements (equivalent to the original PyTorch expression)
    distance = np.sum(x != y)

    return float(distance)

reducer = umap.UMAP(metric = custom_distance, random_state = 295)

# Perform UMAP decomposition
# embed = reducer.fit_transform(stacked_windows)

# plt.figure()
# plt.scatter(embed[:,0], embed[:,1], c = simple_tweetyclr.mean_colors_per_minispec)
# plt.suptitle("UMAP Decomposition of Spectrograms Using Edit Distance Similarity")
# plt.title(f'Number of Slices: {embed.shape[0]}')
# plt.savefig(f'{simple_tweetyclr.folder_name}/Plots/umap_decomp_of_spec_slice_hamming.png')
# plt.show()


# Set up a base dataloader (which we won't directly use for modeling). Also define the batch size of interest
total_dataset = TensorDataset(torch.tensor(simple_tweetyclr.stacked_windows.reshape(simple_tweetyclr.stacked_windows.shape[0], 1, 100, 151)))
batch_size = 72
total_dataloader = DataLoader(total_dataset, batch_size=batch_size , shuffle=False)

# Creating the hover images in the Bokeh plot
list_of_images = []
for batch_idx, (data) in enumerate(total_dataloader):
    data = data[0]

    for image in data:
        list_of_images.append(image)

list_of_images = [tensor.numpy() for tensor in list_of_images]

embeddable_images = simple_tweetyclr.get_images(list_of_images)

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
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau

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
        # self.relu = nn.LeakyReLU(0.01)
        self.sigmoid = nn.Sigmoid()
        
        self.fc1 = nn.Linear(320, 1)
        self.fc2 = nn.Linear(100, 50)

        self._to_linear = 1
        
        
        # Initialize convolutional layers with He initialization
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.zeros_(m.bias)

    def forward_once(self, x):

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

        x = self.relu(self.fc1(x_flattened))
        # Define the clip value
        # clip_value = window_size  # for example
        
        # Apply ReLU and then clip
        # clipped_output = torch.clamp(self.relu(x), max=clip_value)
        # x = self.relu(self.fc2(x))
        # x = self.sigmoid(self.fc1(x_flattened))*100
        
        # return clipped_output
        return x
    
    def forward(self, input1, input2):
        
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        
        return output1, output2
    
    def calculate_loss(self, output1, output2):
        '''

        Parameters
        ----------
        output1 : torch tensor
            Model output (forward pass) of the first spectrogram slice.
        output2 : torch tensor
            Model output (forward pass) of the second spectrogram slice.

        Returns
        -------
        predicted_edit_distance : torch tensor
            Predicted edit distance for each sample in the batch .

        '''
        predicted_edit_distance = torch.abs(output1 - output2).sum(dim = 1).to(torch.float32) 
        return predicted_edit_distance
 
# Initialize random seeds for reproducibility. I chose 295 because it's my favorite highway in NJ
simple_tweetyclr.shuffling(295) # Helper function that randomizes the spectrogram slices 
torch.manual_seed(295)

# Define the shuffled indices. 
shuffled_indices = simple_tweetyclr.shuffled_indices


# The below two lines are used for the "a priori" shuffling procedure. The model is able to train with this "a priori" shuffling procedure 
# stacked_windows = simple_tweetyclr.stacked_windows[shuffled_indices,:]
# labels = simple_tweetyclr.stacked_labels_for_window[shuffled_indices, :]

# Redefining the data 
stacked_windows = simple_tweetyclr.stacked_windows.copy()
labels = simple_tweetyclr.stacked_labels_for_window.copy()

# =============================================================================
# Dataset curation for Siamese Network
# =============================================================================

# I want to create pairs of data observations from a dataset. 
from torch.utils.data import Dataset

class SiameseDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # Just return a single data item
        return self.data[index]

# class SiameseDataset(Dataset):
#     def __init__(self, data, seed = 295):
#         self.data = data
#         self.seed = seed
        
#     def calculate_edit_distance(self, x1, x2):
#         '''

#         Parameters
#         ----------
#         x1 : torch tensor
#             Array of ground truth labels for the first spectrogram slice.
#         x2 : torch tensor
#             Array of ground truth labels for the second spectrogram slice.

#         Returns
#         -------
#         dist : torch tensor
#             Scalar value of the edit distance.

#         '''
#         label1 = x1[1]
#         label2 = x2[1]
        
#         dist = (label1 != label2).sum().unsqueeze(0).to(torch.float32).sum()
#         return dist
        

#     def __getitem__(self, index):
#         random.seed(index)
#         # Select the first item of the pair
#         x1 = self.data[index]

#         # Randomly select the second item of the pair
#         idx2 = random.randint(0, len(self.data) - 1) # Random selection w.r.t. the random seed
#         x2 = self.data[idx2]
        
#         dist = self.calculate_edit_distance(x1, x2)

#         return x1, x2, dist, [index, idx2]

#     def __len__(self):
#         return len(self.data)

# Creating the necessary structure for the dataset for analysis
dataset = torch.tensor(stacked_windows.reshape(stacked_windows.shape[0], 1, 100, 151))

dataset = TensorDataset(dataset, torch.tensor(labels))

# Initialize Dataset and DataLoader
siamese_dataset = SiameseDataset(dataset)

# siamese_dataset = SiameseDataset(dataset)
# a = next(iter(siamese_dataset))
# len(a) = 2, where the first dimension is a tuple of the spectrogram slices and the second dimension is a tuple of the ground truth labels

# Define the split sizes -- what is the train test split ? 
train_perc = 0.8 #
train_size = int(train_perc * len(dataset))  # (100*train_perc)% for training
test_size = len(dataset) - train_size  # 100 - (100*train_perc)% for testing


from torch.utils.data import random_split
# Split the dataset into a training and testing dataset
train_dataset, test_dataset = random_split(siamese_dataset, [train_size, test_size])

shuffle_status = False # Dynamic shuffling within the dataloader 

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_status)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle_status)

a = next(iter(train_loader))
# =============================================================================
# Model Training
# =============================================================================

model = Encoder()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.MSELoss()  # MSE Loss
# scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
num_epochs = 5000
# scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.005, patience=50, verbose=True)

predictions = []
epoch_loss_train = []
epoch_loss_test = []
predicted_vals = []
actual_vals = []

# Set the accumulation steps
accumulation_steps = 1 # This is the number of steps over which you accumulate gradients

# Training Loop
model.train()
for epoch in range(num_epochs):
    batch_loss_train = 0
    for data in train_loader:
        model.train()  # Set model to training mode
        optimizer.zero_grad()
        
        num_samples = len(data[0])
        
        # If odd number of samples, reduce by one to make it even
        if num_samples % 2 != 0:
            num_samples -= 1
        
        # Generate unique indices for the adjusted batch
        unique_indices = np.random.choice(num_samples, size=num_samples, replace=False)
        idx1, idx2 = np.array_split(unique_indices, 2)
        
        x1_inputs = data[0][idx1]
        x1_labels = data[1][idx1]
        x2_inputs = data[0][idx2]
        x2_labels = data[1][idx2]
        
        # Calculate edit distance
        dist = torch.sum((x1_labels != x2_labels), dim=1).unsqueeze(0).to(torch.float32)
        # Transfer inputs and labels to the device
        input1 = torch.stack(list(x1_inputs)).to(device, dtype=torch.float32)
        label1 = torch.stack(list(x1_labels)).to(device, dtype=torch.float32)
        input2 = torch.stack(list(x2_inputs)).to(device, dtype=torch.float32)
        label2 = torch.stack(list(x2_labels)).to(device, dtype=torch.float32)
        targets = dist.to(device).squeeze()
        
        # Pass our two spectrogram slices through the model
        output1, output2 = model(input1, input2)
        
        predicted_edit_distance = model.calculate_loss(output1, output2)

        # predicted_vals.append(predicted_loss.item())
        # actual_vals.append(actual_loss.item())
        
        train_loss = criterion(predicted_edit_distance, targets)
        train_loss.backward()
        batch_loss_train += train_loss.item()        
        optimizer.step()
        
    epoch_loss_train.append(batch_loss_train)
    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_loss_train[-1]:.4f}')



        # # Generate pairs on-the-fly
        # batch_pairs = []
        # for i in range(1, data[0].shape[0] // 2):  # Assuming the batch size is even
        #     idx1 = random.randint(0, (data[0].shape[0]) - 1)
        #     idx2 = random.randint(0, (data[0].shape[0]) - 1)

        #     x1 = [data[0][idx1], data[1][idx1]]
        #     x2 = [data[0][idx2], data[1][idx2]]
            
        #     # Calculate edit distance
        #     dist = (x1[1] != x2[1]).sum().unsqueeze(0).to(torch.float32).sum()

        #     # Optionally, you can also create and include the symmetric pair (x2, x1)
        #     # batch_pairs.append((x1, x2))
        #     batch_pairs.append([x1, x2, dist, [idx1, idx2]])
            
        # for j in np.arange(len(batch_pairs)):
        #     input1, label1 = batch_pairs[j][0][0], batch_pairs[j][0][1]
        #     input2, label2 = batch_pairs[j][1][0], batch_pairs[j][1][1]
            
        #     targets = batch_pairs[j][2]
        
        
        

        
        
        #     input1 = input1.to(device, dtype = torch.float32)
        #     label1 = label1.to(device, dtype = torch.float32)
            
        #     input2 = input2.to(device, dtype = torch.float32)
        #     label2 = label2.to(device, dtype = torch.float32)
            
        #     targets = targets.to(device)
            
            
            




































# for epoch in range(num_epochs):
#     batch_loss_train = 0
#     batch_loss_test = 0
    
#     for i, data in enumerate(train_loader):
#         model.train()  # Set model to training mode
#         optimizer.zero_grad()
        
#         # Transfer inputs and labels to the GPU
#         x1, x2, targets, indices = data
        
#         input1, label1 = x1[0], x1[1]
#         input2, label2 = x2[0], x2[1]
        
        
#         input1 = input1.to(device, dtype = torch.float32)
#         label1 = label1.to(device, dtype = torch.float32)
        
#         input2 = input2.to(device, dtype = torch.float32)
#         label2 = label2.to(device, dtype = torch.float32)
        
#         targets = targets.to(device)
        
#         # Pass our two spectrogram slices through the model
#         output1, output2 = model(input1, input2)
        
        
#         predicted_edit_distance = model.calculate_loss(output1, output2)
#         # predicted_vals.append(predicted_loss.item())
#         # actual_vals.append(actual_loss.item())
        
#         train_loss = criterion(predicted_edit_distance, targets)
#         train_loss.backward()
#         batch_loss_train += train_loss.item()        
#         optimizer.step()
        
#     model.eval()
        
#     for data in test_loader:
        
#         x1, x2, targets, indices = data
        
#         input1, label1 = x1[0], x1[1]
#         input2, label2 = x2[0], x2[1]
        
#         input1 = input1.to(device, dtype = torch.float32)
#         label1 = label1.to(device, dtype = torch.float32)
        
#         input2 = input2.to(device, dtype = torch.float32)
#         label2 = label2.to(device, dtype = torch.float32)
        
#         targets = targets.to(device)

        
#         output1, output2 = model(input1, input2)
        
#         predicted_edit_distance = model.calculate_loss(output1, output2)
#         # predicted_vals.append(predicted_loss.item())
#         # actual_vals.append(actual_loss.item())
        
#         loss = criterion(predicted_edit_distance, targets)
#         loss = loss / accumulation_steps
#         batch_loss_test += loss.item()
        
    
#     # for i, (inputs, labels) in enumerate(test_loader):
#     #     inputs = inputs.to(device, dtype=torch.float32)
#     #     labels = labels.to(device, dtype=torch.float32)
        
#     #     pred_hamming = model(inputs)
        
#     #     actual_hamming = [(obs2 != obs1).sum().unsqueeze(0) for obs1, obs2 in itertools.combinations(labels, 2)]
#     #     actual_hamming = torch.cat(actual_hamming, dim = 0)/100
#     #     actual_hamming = torch.sum(actual_hamming).to(torch.float32)
        
#     #     # actual_hamming = (labels[0,:] != labels[1,:]).sum().to(torch.float32)
#     #     # actual_hamming = actual_hamming.view(1, 1)
#     #     loss = criterion(pred_hamming, actual_hamming)
        
#     #     # Scales the loss, and calls backward() to accumulate gradients
#     #     loss = loss / accumulation_steps
        
#     #     batch_loss_test += loss.item()
        
#     #     break
        
#     # Logging the loss averaged over an epoch
#     epoch_loss_train.append(batch_loss_train)
#     epoch_loss_test.append(batch_loss_test)
#     # Step the scheduler based on validation loss
#     # scheduler.step(train_loss)
    
#     # Assuming 'model' is your neural network model
#     # for name, parameter in model.named_parameters():
#     #     if parameter.grad is not None:
#     #         grad_norm = parameter.grad.norm()
#     #         print(f"Gradient norm of {name}: {grad_norm}")
    
#     # Step the scheduler at the end of the epoch
#     # epoch_loss_train.append(batch_loss_train/len(train_loader))
#     # epoch_loss_test.append(batch_loss_test/len(test_loader))
#     # epoch_loss.append(batch_loss / len(train_loader))
    
#     # print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_loss_train[-1]:.4f}, Test Loss: {epoch_loss_test[-1]:.4f}')
#     print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_loss_train[-1]:.4f}')
    

# epoch_loss_train_arr = np.array(epoch_loss_train)
# epoch_loss_test_arr = np.array(epoch_loss_test)

# plt.figure()
# plt.plot(np.log(epoch_loss_train_arr + 1e-5), label = 'Train Loss')
# plt.plot(np.log(epoch_loss_test_arr + 1e-5), label = 'Validation Loss')
# plt.axhline(np.log(1e-5), color = 'red', linestyle = '--', label = 'Lowest Possible Loss')
# plt.legend()
# plt.title("Loss Curve")
# plt.xlabel("Epoch")
# plt.ylabel("log(loss + 1e-5)")
# plt.savefig(f'{folder_name}/loss_curve.png')
# plt.show()

# # Let's save the parameters from this experiment (this will go in the utils
# # file later)

# # Here are the parameters I want to save:
# # 1. Data parameters: 
#     # a. Number of spectrograms for analysis
#     # b. Stride
#     # c. Window size
#     # d. Total number of spectrogram slices
#     # e. Whether any standardization was applied

# # 2. Model parameters: 
#     # a. Optimizer type and Learning rate
#     # b. Batch size 
#     # c. Number of epochs
#     # d. Random seed
#     # e. Accumulation? 
#     # f. Train/Test split? If so, what is the proportion?
#     # g. Model architecture & any regularlization. 
#     # h. Criterion
    
# # data_params = {
# #     "Data_Directory": bird_dir,
# #     "Window_Size": simple_tweetyclr.window_size, 
# #     "Stride_Size": simple_tweetyclr.stride, 
# #     "Num_Spectrograms": simple_tweetyclr.num_spec, 
# #     "Total_Slices": simple_tweetyclr.stacked_windows.shape[0], 
# #     "Frequencies_of_Interest": masking_freq_tuple, 
# #     "Data_Standardization": "None"
# #     }

# model_arch = str(model)
# forward_method = inspect.getsource(model.forward)
# forward_once_method = inspect.getsource(model.forward_once)

# # Splitting the string into an array of lines
# model_arch_lines = model_arch.split('\n')
# forward_method_lines = forward_method.split('\n')
# forward_once_method_lines = forward_once_method.split('\n')

# experiment_params = {
#     "Data_Directory": bird_dir,
#     "Window_Size": simple_tweetyclr.window_size, 
#     "Stride_Size": simple_tweetyclr.stride, 
#     "Num_Spectrograms": simple_tweetyclr.num_spec, 
#     "Total_Slices": simple_tweetyclr.stacked_windows.shape[0], 
#     "Frequencies_of_Interest": masking_freq_tuple, 
#     "Data_Standardization": "None",
#     "Optimizer": str(optimizer), 
#     "Batch_Size": batch_size, 
#     "Num_Epochs": num_epochs, 
#     "Torch_Random_Seed": 295, 
#     "Accumulation_Size": 1, 
#     "Train_Proportion": train_perc,
#     "Criterion": str(criterion), 
#     "Model_Architecture": model_arch_lines, 
#     "Forward_Method": forward_method_lines, 
#     "Forward_Once_Method": forward_once_method_lines,
#     "Dataloader_Shuffle": shuffle_status
#     }

# import json

# with open(f'{simple_tweetyclr.folder_name}/experiment_params.json', 'w') as file:
#     json.dump(experiment_params, file, indent=4)

# # model_rep = []

# # model.eval()
# # with torch.no_grad():
# #     for i, (inputs, labels) in enumerate(total_dataloader):
# #         inputs = inputs.to(device, dtype=torch.float32)
# #         labels = labels.to(device, dtype=torch.float32)
    
# #         x = F.relu(model.bn1(model.conv1(inputs)))
# #         x = F.relu(model.bn2(model.conv2(x)))
# #         x = F.relu(model.bn3(model.conv3(x)))
# #         x = F.relu(model.bn4(model.conv4(x)))
# #         x = F.relu(model.bn5(model.conv5(x)))
# #         x = F.relu(model.bn6(model.conv6(x)))
# #         x = F.relu(model.bn7(model.conv7(x)))
# #         x = F.relu(model.bn8(model.conv8(x)))
# #         x = F.relu(model.bn9(model.conv9(x)))
# #         x = F.relu(model.bn10(model.conv10(x)))

# #         x_flattened = x.view(-1, 320)
# #         model_rep.append(x_flattened.clone().detach().numpy())
        
# # model_rep1 = np.concatenate((model_rep))    
        
# # Write the experiment information to a txt file 







