#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 10:22:09 2023

@author: akapoor
"""
import numpy as np
import torch
import sys
filepath = '/Users/AnanyaKapoor'
# sys.path.append('/home/akapoor/Dropbox (University of Oregon)/Kapoor_Ananya/01_Projects/01_b_Canary_SSL/TweetyCLR/')
from util import MetricMonitor, SupConLoss
from util import Tweetyclr, Temporal_Augmentation, TwoCropTransform, Custom_Contrastive_Dataset
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
        # self.conv11 = nn.Conv2d(16, 8, 3, 1, padding = 1)
        # self.conv12 = nn.Conv2d(8, 8, 3, 2, padding = 1)
        
        
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
        # self.bn11 = nn.BatchNorm2d(16)
        # self.bn12 = nn.BatchNorm2d(8)

        self.relu = nn.ReLU()       
        self.dropout = nn.Dropout2d(
        )
        # self.fc = nn.Linear(320, 32)
        # self._to_linear = 1280
        # self._to_linear = 320
        self._to_linear = 320
        
    def forward(self, x):
         
        # x = F.relu(self.dropout(self.conv1(self.bn1(x))))
        # x = F.relu(self.conv2(self.bn2(x))) 
        # x = F.relu(self.dropout(self.conv3(self.bn3(x))))
        # x = F.relu(self.conv4(self.bn4(x))) 
        # x = F.relu(self.dropout(self.conv5(self.bn5(x))))
        # x = F.relu(self.conv6(self.bn6(x))) 
        # x = F.relu(self.dropout(self.conv7(self.bn7(x))))
        # x = F.relu(self.conv8(self.bn8(x)))
        # x = F.relu(self.dropout(self.conv9(self.bn9(x))))
        # x = F.relu(self.conv10(self.bn10(x)))
        
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
        # x = F.relu(self.conv11(self.bn11(x)))
        # x = F.relu(self.conv12(self.bn12(x)))

        # x = x.view(-1, 48)
        x = x.view(-1, 320)
        # x = self.fc(x)
        # x = self.relu(x)
        # x = x.view(-1, 32)
        # x = x.view(-1, 1280) #Window size = 500
        
        return x


def pretraining(epoch, model, contrastive_loader, optimizer, criterion, method='SimCLR'):
    "Contrastive pre-training over an epoch. Adapted from XX"
    negative_similarities_for_epoch = []
    ntxent_positive_similarities_for_epoch = []
    mean_pos_cos_sim_for_epoch = []
    mean_ntxent_positive_similarities_for_epoch = []
    metric_monitor = MetricMonitor()
    model.train()
    
    # dummy_dataloader = contrastive_loader[0] # THis will be used for the indices to enumerate
    
    # Use a list comprehension to concatenate data tensors within each tuple along dimension 0
    # data =  [torch.cat([data for data, _ in data_loader], dim=0) for data_loader in contrastive_loader]
    
    # # Concatenate all the data tensors from the list into one tensor
    # data = torch.cat(data, dim=0)


    for batch_data in enumerate(contrastive_loader):
        data_list = []
        label_list = []
        a = batch_data[1]
        for idx in np.arange(len(a)):
            data_list.append(a[idx][0])
        
        
    # for batch_idx, ((data1, labels1), (data2, labels2)) in enumerate(contrastive_loader):
        data = torch.cat((data_list), dim = 0)
        data = data.unsqueeze(1)
        # data = data.reshape(a[idx][0].shape[0], len(a), a[idx][0].shape[1], a[idx][0].shape[2])
        # labels = labels1
        if torch.cuda.is_available():
            data = data.cuda()
        data = torch.autograd.Variable(data,False)
        bsz = a[idx][0].shape[0]
        data = data.to(torch.float32)
        features = model(data)
        norm = features.norm(p=2, dim=1, keepdim=True)
        epsilon = 1e-12
        # Add a small epsilon to prevent division by zero
        normalized_tensor = features / (norm + epsilon)
        split_features = torch.split(normalized_tensor, [bsz]*len(a), dim=0)
        split_features = [split.unsqueeze(1) for split in split_features]

        features = torch.cat(split_features, dim = 1)

        loss, negative_similarities, positive_similarities = criterion(features)

        metric_monitor.update("Loss", loss.item())
        metric_monitor.update("Learning Rate", optimizer.param_groups[0]['lr'])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        negative_similarities_for_epoch.append(float(np.mean(negative_similarities.clone().detach().cpu().numpy())))
        ntxent_positive_similarities_for_epoch.append(float(np.mean(positive_similarities.clone().detach().cpu().numpy())))
        
        # # Calculate the mean cosine similarity of the model feature representation for the positive pairs.
        # # Slice the tensor to separate the two sets of features you want to compare
        
        split_features = torch.split(normalized_tensor, [bsz]*len(a), dim=0)
        split_features = [split.unsqueeze(1) for split in split_features]
        features = torch.cat(split_features, dim = 1)
        
        
        # f1, f2 = torch.split(normalized_tensor, [bsz, bsz], dim=0)
        # normalized_features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        
        # tensor_a = normalized_features[:, 0, :].clone().detach()
        # tensor_b = normalized_features[:, 1, :].clone().detach()
        
        # # Compute the cosine similarities
        # similarities = F.cosine_similarity(tensor_a, tensor_b, dim=1)
        # mean_pos_cos_sim_for_batch = torch.mean(similarities).clone().detach().cpu().numpy()
        # mean_ntxent_positive_similarities = torch.mean(positive_similarities).clone().detach().cpu().numpy()
        # mean_ntxent_positive_similarities_for_epoch.append(float(mean_ntxent_positive_similarities))
        # mean_pos_cos_sim_for_epoch.append(float(mean_pos_cos_sim_for_batch))

    print("[Epoch: {epoch:03d}] Contrastive Pre-train | {metric_monitor}".format(epoch=epoch, metric_monitor=metric_monitor))
    # return metric_monitor.metrics['Loss']['avg'], metric_monitor.metrics['Learning Rate']['avg'], negative_similarities_for_epoch, features, mean_pos_cos_sim_for_epoch, mean_ntxent_positive_similarities_for_epoch
    return metric_monitor.metrics['Loss']['avg'], metric_monitor.metrics['Learning Rate']['avg'], features, negative_similarities_for_epoch, ntxent_positive_similarities_for_epoch


# =============================================================================
#     # Set data parameters
# =============================================================================

# First let's validate on just one bird and for pretransection data

analysis_path = f'{filepath}/Dropbox (University of Oregon)/Kapoor_Ananya/01_Projects/01_b_Canary_SSL/TweetyCLR_Structure_Repo/'
bird_dir = f'{analysis_path}Rose_Canary_Python_Files/14'

# Parameters we set
num_spec = 20
window_size = 100
stride = 10

# Define the folder name
folder_name = f'{analysis_path}Num_Spectrograms_{num_spec}_Window_Size_{window_size}_Stride_{stride}'

files = os.listdir(f'{bird_dir}/song')
all_songs_data = [f'{bird_dir}/song/{element}' for element in files if '.npz' in element] # Get the file paths of each numpy file from Yarden's data
all_songs_data.sort()

masking_freq_tuple = (500, 7000)
spec_dim_tuple = (window_size, 151)


exclude_transitions = False


# =============================================================================
#     # Set model parameters
# =============================================================================

batch_size = 3
num_epochs = 10
tau_in_steps = 3
temp_value = 0.02
method = 'SimCLR'
device = 'cpu'
use_scheduler = True

# =============================================================================
#     # Initialize the TweetyCLR object
# =============================================================================
simple_tweetyclr_experiment_1 = Tweetyclr(num_spec, window_size, stride, folder_name, all_songs_data, masking_freq_tuple, spec_dim_tuple, exclude_transitions)
simple_tweetyclr_experiment_1.temperature_value = 0.02
simple_tweetyclr = simple_tweetyclr_experiment_1

# =============================================================================
#     # Data processing
# =============================================================================

simple_tweetyclr.first_time_analysis()

# for i in np.arange(100):
#     plt.figure()
#     plt.imshow(simple_tweetyclr.stacked_windows[i,:].reshape(100,151), cmap = 'jet')
#     plt.show()

# dx = simple_tweetyclr.stacked_window_times[0,1] - simple_tweetyclr.stacked_window_times[0,0] 

# stacked_times = dx*np.arange(simple_tweetyclr.stacked_specs.shape[1])
# stacked_times = stacked_times.reshape(1, stacked_times.shape[0])

# spec = simple_tweetyclr.stacked_specs

# # I want to load one spectrogram just to see what it looks like 
# dat = np.load('/Users/AnanyaKapoor/Dropbox (University of Oregon)/Kapoor_Ananya/01_Projects/01_b_Canary_SSL/TweetyCLR_Structure_Repo/Rose_Canary_Python_Files/14/song/USA5207_45107_60380709_6_30_16_46_20.npz')
# # t = dat['t']
# f = dat['f']
# # s = dat['s']
# # Let's get rid of higher order frequencies
# mask = (f<7000)&(f>500)
# masked_frequencies = f[mask].reshape(151,1)

# # spec = dat['s']

# plt.figure()
# plt.pcolormesh(stacked_times, masked_frequencies, spec, cmap = 'jet')
# plt.show()

# I want to stack the first 100 spectrogram slices together
# spec_list_temp = []

# for i in np.arange(100):
#     spec_list_temp.append(simple_tweetyclr.stacked_windows[i,:].reshape(100,151))


# spec_list_arr = np.concatenate((spec_list_temp), axis = 0).T

# dx = simple_tweetyclr.stacked_window_times[0,1] - simple_tweetyclr.stacked_window_times[0,0] 

# stacked_times = dx*np.arange(spec_list_arr.shape[1])
# stacked_times = stacked_times.reshape(1, stacked_times.shape[0])

# dat = np.load('/Users/AnanyaKapoor/Dropbox (University of Oregon)/Kapoor_Ananya/01_Projects/01_b_Canary_SSL/TweetyCLR_Structure_Repo/Rose_Canary_Python_Files/14/song/USA5207_45107_60380709_6_30_16_46_20.npz')
# # # t = dat['t']
# f = dat['f']
# # # s = dat['s']
# # # Let's get rid of higher order frequencies
# mask = (f<7000)&(f>500)
# masked_frequencies = f[mask].reshape(151,1)

# plt.figure()
# plt.pcolormesh(stacked_times, masked_frequencies, spec_list_arr, cmap = 'jet')
# plt.show()


simple_tweetyclr.shuffling(295)
# shuffled_indices = np.load(f'{simple_tweetyclr.folder_name}/shuffled_indices.npy')
# simple_tweetyclr.shuffling(shuffled_indices)
shuffled_indices = simple_tweetyclr.shuffled_indices    

# =============================================================================
#     # What data do we want to use? Raw? Log? Z-Score? 
# =============================================================================

# Raw
data_for_analysis = simple_tweetyclr.stacked_windows 

# =============================================================================
#     # Create training dataset and visualization dataloader.
# =============================================================================

# # # Create training and testing data
shuffled_indices = simple_tweetyclr.shuffled_indices
train_perc = 1.0

stacked_windows_train = simple_tweetyclr.stacked_windows
stacked_windows_train.shape = (stacked_windows_train.shape[0], 1, 100, 151)
anchor_indices = shuffled_indices

# stacked_windows_train, anchor_indices  = simple_tweetyclr.train_test_split(data_for_analysis, train_perc, shuffled_indices)

train_indices = torch.tensor(anchor_indices, dtype=torch.long)

# train_dataset = TensorDataset(stacked_windows_train, train_indices)

# # Create a DataLoader
# simple_tweetyclr.train_dataloader = DataLoader(train_dataset, batch_size=batch_size , shuffle=False)

total_dataset = TensorDataset(torch.tensor(data_for_analysis.reshape(data_for_analysis.shape[0], 1, simple_tweetyclr.time_dim, simple_tweetyclr.freq_dim)))
total_dataloader = DataLoader(total_dataset, batch_size=batch_size , shuffle=False)

list_of_images = []
for batch_idx, (data) in enumerate(total_dataloader):
    data = data[0]
    
    for image in data:
        list_of_images.append(image)
        
list_of_images = [tensor.numpy() for tensor in list_of_images]

embeddable_images = simple_tweetyclr.get_images(list_of_images)

# =============================================================================
#     # Create a dictionary that we will use to create augmentations
# =============================================================================

# Shuffle and Subset the dictionary

# Convert dictionary to a list of key-value pairs
dict_of_spec_slices_with_slice_number = {i: data_for_analysis[i, :] for i in range(data_for_analysis.shape[0])}

items = list(dict_of_spec_slices_with_slice_number.items())

# Use indices to reorder items
shuffled_items = [items[i] for i in shuffled_indices]

# Convert reordered list of key-value pairs back to a dictionary
simple_tweetyclr.dict_of_spec_slices_with_slice_number = dict(shuffled_items)

# =============================================================================
#     # Create contrastive dataloaders
# =============================================================================

augmentation_object = Temporal_Augmentation(simple_tweetyclr.dict_of_spec_slices_with_slice_number , simple_tweetyclr, tau_in_steps=tau_in_steps)

custom_transformation = TwoCropTransform(augmentation_object)

# # Your new_contrastive_dataset initialization would be:
new_contrastive_dataset = Custom_Contrastive_Dataset(stacked_windows_train, train_indices, custom_transformation)

# # DataLoader remains the same
contrastive_loader = torch.utils.data.DataLoader(new_contrastive_dataset, batch_size=batch_size, shuffle=False)

# Let's plot one anchor slice with its two positive augmentations
a = next(iter(contrastive_loader))
for i in np.arange(3):
    plt.figure()
    plt.imshow(a[0][i][0].clone().detach().numpy().squeeze(), cmap = 'jet')
    plt.title(f'Augmentation {i}')
    plt.show()

# =============================================================================
#     # Let's do some local saving to save on computational time
# =============================================================================

aug_tensor = torch.empty((tau_in_steps, 0, 1, simple_tweetyclr.window_size, 151))

# first_aug_tensor = torch.empty((0, 1, simple_tweetyclr.window_size, 151))
# second_aug_tensor = torch.empty((0, 1, simple_tweetyclr.window_size, 151))
labels_tensor = torch.empty((0))

aug_dict = {}

# Iterate through a range of 15 keys
for i in range(tau_in_steps):
    value = []  # Initialize an empty list as the value for each key
    aug_dict[i] = value

# aug_list = []
labels_list = []
batch_sizes = []

for batch_idx, (data, labels) in enumerate(contrastive_loader):
    labels_list.append(labels)
    
    for i in np.arange(len(data)):
        aug = data[i]
        # aug_tensor[i,:,:,:] = aug
        aug_dict[i].append(aug)
    
labels_tensor = torch.cat(labels_list, dim=0)

flattened_dict = {key: [item for sublist in value for item in sublist] for key, value in aug_dict.items()}

# Initialize a list to store the dictionaries
dataloader_list = []
# filepath_list = []



for i in np.arange(len(flattened_dict)):
    
    aug_data = torch.cat(flattened_dict[i], dim = 0)
    dataset = TensorDataset(aug_data, labels_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    dataloader_list.append(dataloader)
    
# =============================================================================
#     # UMAP on data_for_analysis as a base unsupervised model
# =============================================================================

# data_tensor = torch.cat([batch[0] for batch in total_dataloader])

# # Raw UMAP on the original spectrogram slices

# reducer = umap.UMAP(metric = 'cosine')

# a = data_tensor.reshape(data_tensor.shape[0], data_tensor.shape[2]*data_tensor.shape[3]).clone().detach().numpy()
# embed = reducer.fit_transform(a)

# plt.figure()
# plt.title("Raw UMAP on Raw Spectrograms")
# plt.scatter(embed[:,0], embed[:,1], c = simple_tweetyclr.mean_colors_per_minispec)
# plt.savefig(f'{simple_tweetyclr.folder_name}/Plots/UMAP_of_raw_spectrograms.png')
# plt.show()

# simple_tweetyclr.plot_UMAP_embedding(embed, simple_tweetyclr.mean_colors_per_minispec, embeddable_images, f'{simple_tweetyclr.folder_name}/Plots/UMAP_of_raw_spectrograms.html', saveflag = True)




# =============================================================================
#     # Pass data through untrained model and extract representation
# =============================================================================

# Untrained Model Representation
# Ensure the model is on the desired device
model = Encoder().to(torch.float32).to(device)
# if torch.cuda.is_available():
#     model = model.cuda()
    # criterion = criterion.cuda()   
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-08)
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)  # Using weight decay with AdamW
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9)

# model = model.to('cpu')
# model.eval()

# # Initialize lists to store features and labels
# model_rep_list_untrained = []

# # Iterate over the DataLoaders
# with torch.no_grad():  # Disable gradient computation for efficiency
#     # for data_loader in dataloader_list:
#     for batch_idx, (data) in enumerate(total_dataloader):
#         data = data[0].to(torch.float32)
#         features = model(data)
#         model_rep_list_untrained.append(features)

# # Convert lists to tensors
# model_rep_untrained = torch.cat(model_rep_list_untrained, dim=0)

# #  UMAP on the untrained model
# reducer = umap.UMAP(metric = 'cosine')

# a = model_rep_untrained.clone().detach().numpy()
# embed = reducer.fit_transform(a)

# # I want my mean_colors_per_minispec to be colorized by time
# import matplotlib.colors as mcolors
# from matplotlib.cm import ScalarMappable

# # Generate hues based on row number
# hues = np.linspace(0, 1, embed.shape[0], endpoint=False)

# # Create an HSV array (saturation and value are both set to 1)
# hsv_colors = np.column_stack((hues, np.ones_like(hues), np.ones_like(hues)))

# # Convert HSV to RGB
# mean_colors_per_minispec = mcolors.hsv_to_rgb(hsv_colors)


# plt.figure()
# plt.title("Data UMAP Representation Through the Untrained Model")
# plt.scatter(embed[:,0], embed[:,1], c = mean_colors_per_minispec)

# # Create a ScalarMappable and initialize a normalization
# # (In this case, from 0 to k, corresponding to your number of rows)
# sm = ScalarMappable(cmap='hsv', norm=plt.Normalize(0, embed.shape[0]))
# sm.set_array([])  # Only needed for older versions of matplotlib

# # Add the color bar using the ScalarMappable
# cbar = plt.colorbar(sm, ticks=np.linspace(0, embed.shape[0], num=5))  # Adjust ticks if needed
# cbar.set_label('Row Number')

# plt.savefig(f'{simple_tweetyclr.folder_name}/Plots/UMAP_of_untrained_model.png')

# plt.show()

# simple_tweetyclr.plot_UMAP_embedding(embed, mean_colors_per_minispec,embeddable_images, f'{simple_tweetyclr.folder_name}/Plots/UMAP_of_untrained_model.html', saveflag = True)

# =============================================================================
#     # Train the model! Extract the model representation
# =============================================================================
contrastive_loss, contrastive_lr = [], []

model = model.to(device)
model.train()
for epoch in range(1, num_epochs+1):
    criterion = SupConLoss(temperature=temp_value)
    # loss, lr, negative_similarities_for_epoch, features, mean_pos_cos_sim_for_epoch, mean_ntxent_positive_similarities_for_epoch = pretraining(epoch, model, zip(*dataloader_list), optimizer, criterion, method=method)
    loss, lr, features, neg_sims, pos_sims = pretraining(epoch, model, zip(*dataloader_list), optimizer, criterion, method=method)

    if use_scheduler:
        scheduler.step()
            
    contrastive_loss.append(loss)
    contrastive_lr.append(lr)
    
model = model.to('cpu')
model.eval()

# Initialize lists to store features and labels
model_rep_list_trained = []

# Iterate over the DataLoaders
with torch.no_grad():  # Disable gradient computation for efficiency
    # for data_loader in dataloader_list:
    for batch_idx, (data) in enumerate(total_dataloader):
        data = data[0].to(torch.float32)
        features = model(data)
        model_rep_list_trained.append(features)

# Convert lists to tensors
model_rep_trained = torch.cat(model_rep_list_trained, dim=0)

reducer = umap.UMAP(metric='cosine')

# Compute the mean and standard deviation for each row
mean = model_rep_trained.mean(dim=1, keepdim=True)
std = model_rep_trained.std(dim=1, keepdim=True, unbiased=False)

trained_rep_umap = reducer.fit_transform(model_rep_trained.clone().detach().numpy())


color = np.array([0.5, 0, 0.5])
mean_colors_per_minispec = np.tile(color, (trained_rep_umap.shape[0], 1))





plt.figure()
plt.title("Data UMAP Representation Through the Trained Model")
plt.scatter(trained_rep_umap[:,0], trained_rep_umap[:,1], c = mean_colors_per_minispec)
plt.savefig(f'{simple_tweetyclr.folder_name}/UMAP_of_trained_model_epoch_2.png')
plt.show()

simple_tweetyclr.plot_UMAP_embedding(trained_rep_umap,mean_colors_per_minispec, embeddable_images, f'{simple_tweetyclr.folder_name}/Plots/UMAP_of_trained_model.html', saveflag = True)


















