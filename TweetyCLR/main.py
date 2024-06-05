#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 14:40:41 2023

@author: akapoor
"""

import numpy as np
import torch
import sys
sys.path.append('src')
sys.path.append('utils')
# sys.path.append('/Users/AnanyaKapoor/Dropbox (University of Oregon)/Kapoor_Ananya/01_Projects/01_b_Canary_SSL/TweetyCLR/')
from utils import SupConLoss
from utils import Tweetyclr, pretraining
from Augmentation import Temporal_Augmentation
from Contrastive_Dataloaders import TwoCropTransform, Custom_Contrastive_Dataset
import torch.nn as nn
import torch.nn.functional as F
import os
from torch.utils.data import DataLoader, TensorDataset
import umap
import matplotlib.pyplot as plt
import torch.optim as optim

from model import Encoder

plt.rcParams.update({'font.size': 20})
plt.rcParams['figure.figsize'] = [15, 15]  # width and height should be in inches, e.g., [10, 6]

# ===================================================================================
#     # Set data parameters -- This will go in an experiment manager json file later
# ===================================================================================
bird_dir = '/Users/AnanyaKapoor/Dropbox (University of Oregon)/Kapoor_Ananya/01_Projects/01_a_Rotations/Gardner_Lab/Canary_Data/llb16_data_matrices/'
# audio_files = bird_dir+'llb3_songs'
directory = bird_dir+ 'Python_Files'
analysis_path = '/Users/AnanyaKapoor/Dropbox (University of Oregon)/Kapoor_Ananya/01_Projects/01_b_Canary_SSL/TweetyCLR_Repo/'

# Parameters we set
num_spec = 150
window_size = 100
stride = 50

# Define the folder name
folder_name = f'{analysis_path}Num_Spectrograms_{num_spec}_Window_Size_{window_size}_Stride_{stride}'


files = os.listdir(directory)
all_songs_data = [f'{directory}/{element}' for element in files if '.npz' in element] # Get the file paths of each numpy file from Yarden's data
all_songs_data.sort()

masking_freq_tuple = (500, 7000)
spec_dim_tuple = (window_size, 151)


# with open('/Users/AnanyaKapoor/Desktop/category_colors.pkl', 'rb') as file:
#     category_colors = pickle.load(file)
    
exclude_transitions = False


# =============================================================================
#     # Set model parameters
# =============================================================================

easy_negatives = 1
hard_negatives = 1
batch_size = easy_negatives + hard_negatives + 1
num_epochs = 100
tau_in_steps = 3
temp_value = 0.02
method = 'SimCLR'
device = 'cpu'
use_scheduler = True

# =============================================================================
#     # Initialize the TweetyCLR object
# =============================================================================
simple_tweetyclr_experiment_1 = Tweetyclr(num_spec, window_size, stride, folder_name, all_songs_data, masking_freq_tuple, spec_dim_tuple, exclude_transitions, category_colors = None)

simple_tweetyclr = simple_tweetyclr_experiment_1
simple_tweetyclr.first_time_analysis()

simple_tweetyclr.easy_negatives = easy_negatives
simple_tweetyclr.hard_negatives = hard_negatives 



# Let's do raw umap

reducer = umap.UMAP(metric = 'cosine', random_state=295)
# reducer = umap.UMAP(metric = 'cosine')

np.save('/Users/AnanyaKapoor/Desktop/stacked_windows_small.npy', simple_tweetyclr.stacked_windows)





# embed = reducer.fit_transform(simple_tweetyclr.stacked_windows)
simple_tweetyclr.umap_embed_init = embed

embed = np.load('/Users/AnanyaKapoor/Downloads/embed (4).npy')

plt.figure()
plt.scatter(embed[:,0], embed[:,1], s = 10, c = simple_tweetyclr.mean_colors_per_minispec)
plt.show()

plt.figure()
plt.scatter(embed[:,0], embed[:,1], s = 10, c = simple_tweetyclr.mean_colors_per_minispec)
plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
# plt.title("UMAP Decomposition of ")
# plt.suptitle("UMAP Representation of All Spectrogram Slices")
plt.title(f'Total Slices: {embed.shape[0]}')
plt.savefig(f'{simple_tweetyclr.folder_name}/Plots/UMAP_of_all_slices.png')
plt.show()


# save the umap embedding for easy future loading
np.save('/Users/AnanyaKapoor/Desktop/embed.npy', embed)

# DEFINE HARD INDICES THROUGH INTERACTION: USER NEEDS TO ZOOM IN ON ROI 

# Get current axes
ax = plt.gca()
# Get current limits
xlim = ax.get_xlim()
ylim = ax.get_ylim()

hard_indices = np.where((embed[:,0]>=xlim[0])&(embed[:,0]<=xlim[1]) & (embed[:,1]>=ylim[0]) & (embed[:,1]<=ylim[1]))[0]

plt.figure(figsize = (10,10))
plt.scatter(embed[hard_indices,0], embed[hard_indices,1], s = 10, c = simple_tweetyclr.mean_colors_per_minispec[hard_indices,:])
plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
# plt.title("UMAP Decomposition of ")
# plt.suptitle("UMAP Representation of All Spectrogram Slices")
plt.title(f'Total Slices: {embed[hard_indices,:].shape[0]}')
plt.savefig(f'{simple_tweetyclr.folder_name}/Plots/UMAP_of_hard_slices.png')
plt.show()

simple_tweetyclr.hard_indices = hard_indices

dataset = simple_tweetyclr.stacked_windows.copy()

# These will give the training and testing ANCHORS
stacked_windows_train, stacked_labels_train, mean_colors_per_minispec_train, hard_indices_train, stacked_windows_test, stacked_labels_test, mean_colors_per_minispec_test, hard_indices_test = simple_tweetyclr.train_test_split(dataset, 0.8, hard_indices)

data_for_analysis = dataset.copy() 

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

# =============================================================================
# # Hand select negative samples
# =============================================================================

# Some negative samples need to be hard. I will do this by for each sample in 
# the bounding box, selecting spectrogram slices within the bounding box that 
# have a low cosine similarity. 

training_indices, stacked_windows_train = simple_tweetyclr.negative_sample_selection(data_for_analysis, hard_indices_train, None)

# Let's eliminate the easy training negatives from the selection of easy negatives for the testing dataset
all_easy_indices = np.array(list(set(np.arange(data_for_analysis.shape[0])) - set(hard_indices)))
training_easy_negatives = np.intersect1d(training_indices, all_easy_indices)

testing_indices, stacked_windows_test = simple_tweetyclr.negative_sample_selection(data_for_analysis, hard_indices_test, training_easy_negatives)

dict_of_spec_slices_with_slice_number = {i: data_for_analysis[i, :] for i in range(data_for_analysis.shape[0])}

items = list(dict_of_spec_slices_with_slice_number.items())

# =============================================================================
#     # Create contrastive dataloaders
# =============================================================================
    
augmentation_object = Temporal_Augmentation(dict_of_spec_slices_with_slice_number , simple_tweetyclr, tau_in_steps=tau_in_steps)

custom_transformation = TwoCropTransform(augmentation_object)

# Training contrastive loader
# # Your new_contrastive_dataset initialization would be:
new_contrastive_dataset = Custom_Contrastive_Dataset(stacked_windows_train, torch.tensor(training_indices), custom_transformation)

# # DataLoader remains the same
contrastive_loader_train = torch.utils.data.DataLoader(new_contrastive_dataset, batch_size=batch_size, shuffle=False)

# Testing contrastive loader
new_contrastive_dataset = Custom_Contrastive_Dataset(stacked_windows_test, torch.tensor(testing_indices), custom_transformation)

# # DataLoader remains the same
contrastive_loader_test = torch.utils.data.DataLoader(new_contrastive_dataset, batch_size=batch_size, shuffle=False)

# =============================================================================
#     # Let's do some local saving to save on computational time
# =============================================================================

aug_tensor_train = torch.empty((tau_in_steps, 0, 1, simple_tweetyclr.window_size, 151))

# first_aug_tensor = torch.empty((0, 1, simple_tweetyclr.window_size, 151))
# second_aug_tensor = torch.empty((0, 1, simple_tweetyclr.window_size, 151))
labels_tensor_train = torch.empty((0))

aug_dict_train = {}

# Iterate through a range of 15 keys
for i in range(tau_in_steps):
    value = []  # Initialize an empty list as the value for each key
    aug_dict_train[i] = value

# aug_list = []
labels_list_train = []
batch_sizes_train = []

for batch_idx, (data, labels) in enumerate(contrastive_loader_train):
    labels_list_train.append(labels)
    
    for i in np.arange(len(data)):
        aug = data[i]
        # aug_tensor[i,:,:,:] = aug
        aug_dict_train[i].append(aug)
    
labels_tensor_train = torch.cat(labels_list_train, dim=0)

flattened_dict_train = {key: [item for sublist in value for item in sublist] for key, value in aug_dict_train.items()}

# Initialize a list to store the dictionaries
dataloader_list_train = []
# filepath_list = []


for i in np.arange(len(flattened_dict_train)):
    
    aug_data = torch.cat(flattened_dict_train[i], dim = 0)
    dataset = TensorDataset(aug_data, labels_tensor_train)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    dataloader_list_train.append(dataloader)
    
#####
aug_tensor_test = torch.empty((tau_in_steps, 0, 1, simple_tweetyclr.window_size, 151))

# first_aug_tensor = torch.empty((0, 1, simple_tweetyclr.window_size, 151))
# second_aug_tensor = torch.empty((0, 1, simple_tweetyclr.window_size, 151))
labels_tensor_test = torch.empty((0))

aug_dict_test = {}

# Iterate through a range of 15 keys
for i in range(tau_in_steps):
    value = []  # Initialize an empty list as the value for each key
    aug_dict_test[i] = value

# aug_list = []
labels_list_test = []
batch_sizes_test = []

for batch_idx, (data, labels) in enumerate(contrastive_loader_test):
    labels_list_test.append(labels)
    
    for i in np.arange(len(data)):
        aug = data[i]
        # aug_tensor[i,:,:,:] = aug
        aug_dict_test[i].append(aug)
    
labels_tensor_test = torch.cat(labels_list_test, dim=0)

flattened_dict_test = {key: [item for sublist in value for item in sublist] for key, value in aug_dict_test.items()}

# Initialize a list to store the dictionaries
dataloader_list_test = []
# filepath_list = []


for i in np.arange(len(flattened_dict_test)):
    
    aug_data = torch.cat(flattened_dict_test[i], dim = 0)
    dataset = TensorDataset(aug_data, labels_tensor_test)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    dataloader_list_test.append(dataloader)    
    

hard_stacked_windows = simple_tweetyclr.stacked_windows[hard_indices,:]

hard_dataset = TensorDataset(torch.tensor(hard_stacked_windows.reshape(hard_stacked_windows.shape[0], 1, simple_tweetyclr.time_dim, simple_tweetyclr.freq_dim)))
hard_dataloader = DataLoader(hard_dataset, batch_size=batch_size , shuffle=False)



hard_stacked_windows = simple_tweetyclr.stacked_windows[hard_indices_train,:]
hard_dataset_train = TensorDataset(torch.tensor(hard_stacked_windows.reshape(hard_stacked_windows.shape[0], 1, simple_tweetyclr.time_dim, simple_tweetyclr.freq_dim)))
hard_dataloader_train = DataLoader(hard_dataset_train, batch_size=batch_size , shuffle=False)

hard_stacked_windows = simple_tweetyclr.stacked_windows[hard_indices_test,:]
hard_dataset_test = TensorDataset(torch.tensor(hard_stacked_windows.reshape(hard_stacked_windows.shape[0], 1, simple_tweetyclr.time_dim, simple_tweetyclr.freq_dim)))
hard_dataloader_test = DataLoader(hard_dataset_test, batch_size=batch_size , shuffle=False)

list_of_images_train = []
for batch_idx, (data) in enumerate(hard_dataloader_train):
    data = data[0]
    
    for image in data:
        list_of_images_train.append(image)
        
list_of_images_train = [tensor.numpy() for tensor in list_of_images_train]

embeddable_images_train = simple_tweetyclr.get_images(list_of_images_train)


list_of_images_test = []
for batch_idx, (data) in enumerate(hard_dataloader_test):
    data = data[0]
    
    for image in data:
        list_of_images_test.append(image)
        
list_of_images_test = [tensor.numpy() for tensor in list_of_images_test]

embeddable_images_test = simple_tweetyclr.get_images(list_of_images_test)

# =============================================================================
#     # Pass data through untrained model and extract representation
# =============================================================================

# Untrained Model Representation
# Ensure the model is on the desired device
model = Encoder().to(torch.float32)
# Check if multiple GPUs are available
if torch.cuda.device_count() > 1:
    print(f"Number of GPUs available: {torch.cuda.device_count()}")
    model = nn.DataParallel(model)

# Move your model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-08)
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)  # Using weight decay with AdamW
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9)

model = model.to('cpu')
original_model = model.module
original_model.to('cpu')
model.eval()
# model = model.to('cpu')
# Initialize lists to store features and labels
model_rep_list_untrained = []

criterion = SupConLoss(temperature=temp_value)
# Iterate over the DataLoaders
with torch.no_grad():  # Disable gradient computation for efficiency
    # for data_loader in dataloader_list:
    for batch_idx, (data) in enumerate(hard_dataloader_train):
        data = data[0].to(torch.float32)
        # features = original_model(data)
        features = model(data)
        model_rep_list_untrained.append(features)

# Convert lists to tensors
model_rep_untrained = torch.cat(model_rep_list_untrained, dim=0)

#  UMAP on the untrained model
reducer = umap.UMAP(metric = 'cosine')

a = model_rep_untrained.clone().detach().numpy()
embed = reducer.fit_transform(a)

plt.figure()
plt.suptitle("Training UMAP Representation Through the Untrained Model")
plt.title(f'Number of Slices: {hard_indices_train.shape[0]}')
plt.scatter(embed[:,0], embed[:,1], c = simple_tweetyclr.mean_colors_per_minispec[hard_indices_train,:])
plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
plt.savefig(f'{simple_tweetyclr.folder_name}/Plots/UMAP_of_untrained_model.png')
plt.show()

simple_tweetyclr.plot_UMAP_embedding(embed, simple_tweetyclr.mean_colors_per_minispec[hard_indices_train,:],embeddable_images_train, f'{simple_tweetyclr.folder_name}/Plots/UMAP_of_untrained_model.html', saveflag = True)

torch.save(model.state_dict(), f'{simple_tweetyclr.folder_name}/untrained_model_state_dict.pth')

training_contrastive_loss, validation_contrastive_loss, contrastive_lr = [], [], []
model = model.to(device)
if torch.cuda.is_available():
    model = model.cuda()
    criterion = criterion.cuda()   
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-08)
# from torch.optim.lr_scheduler import CosineAnnealingLR
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)  # Using weight decay with AdamW
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9)
# scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0)
model.train()

# THIS BELOW TRAINING LOOP NEEDS TO BE MODIFIED SO THAT I AM EVALUATING THE TRAINING AND VALIDATION PERFORMANCE. ALSO INCLUDE EARLY STOPPING. 

initial_negative_sims = []
initial_positive_sims = []

final_negative_sims = []
final_positive_sims = []


for epoch in range(0, num_epochs+1):
    criterion = SupConLoss(temperature=temp_value)
    # loss, lr, negative_similarities_for_epoch, features, mean_pos_cos_sim_for_epoch, mean_ntxent_positive_similarities_for_epoch = pretraining(epoch, model, zip(*dataloader_list), optimizer, criterion, method=method)
    training_loss, validation_loss, lr, training_features, validation_features, negative_sims, positive_sims = pretraining(epoch, model, zip(*dataloader_list_train), zip(*dataloader_list_test), optimizer, criterion, epoch, method=method)
    
    if epoch == 0:
        initial_negative_sims.append(negative_sims)
        initial_positive_sims.append(positive_sims)
    elif epoch == num_epochs:
        final_negative_sims.append(negative_sims)
        final_positive_sims.append(positive_sims)
        
    if (use_scheduler == True) & (epoch !=0):
        scheduler.step()
            
    training_contrastive_loss.append(training_loss)
    validation_contrastive_loss.append(validation_loss)
    contrastive_lr.append(lr)
    
    if epoch%10 == 0:
        torch.save(model.state_dict(), f'{simple_tweetyclr.folder_name}/_model_state_epoch_{epoch}_dict.pth')

    

# UNTRAINED 

plt.figure()
plt.hist([num * temp_value for num in initial_positive_sims[0]], alpha = 0.7, edgecolor = 'black', color = 'blue', label = 'Positive Sample Similarities')
plt.hist([num * temp_value for num in initial_negative_sims[0]], alpha = 0.7, edgecolor = 'black', color = 'red', label = 'Negative Sample Similarities')
plt.legend()
from scipy.stats import ks_2samp

ks, p = ks_2samp([num * temp_value for num in initial_positive_sims[0]], [num * temp_value for num in initial_negative_sims[0]])
print(ks)
plt.title("Representational Similarities of Untrained Model")
# plt.title(f'KS Value: {ks:.3f}')
plt.xlabel('Similarity Score')
plt.ylabel("Frequency")
plt.show()


# TRAINED

plt.figure()
plt.hist([num * temp_value for num in final_positive_sims[0]], alpha = 0.7, edgecolor = 'black', color = 'blue', label = 'Positive Sample Similarities')
plt.hist([num * temp_value for num in final_negative_sims[0]], alpha = 0.7, edgecolor = 'black', color = 'red', label = 'Negative Sample Similarities')
plt.legend()
from scipy.stats import ks_2samp

ks, p = ks_2samp([num * temp_value for num in final_positive_sims[0]], [num * temp_value for num in final_negative_sims[0]])
print(ks)
plt.title("Representational Similarities of Trained Model")
# plt.title(f'KS Value: {ks:.3f}')
plt.xlabel('Similarity Score')
plt.ylabel("Frequency")
plt.show()





# # Let's plot the similarity histograms before and after training
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))  # 1 row, 2 columns

# ax1.hist([num * temp_value for num in initial_negative_sims[0]])
# ax1.set_title("Negative Similarity Scores for all Batches After First Epoch")
# ax1.set_xlabel("Similarity Score")
# ax1.set_ylabel('Frequency')

# ax2.hist([num * temp_value for num in initial_positive_sims[0]])
# ax2.set_title("Positive Similarlity Scores for all Batches After First Epoch")
# ax2.set_xlabel("Similarity Score")
# ax2.set_ylabel('Frequency')

# plt.show()

# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))  # 1 row, 2 columns

# ax1.hist([num * temp_value for num in final_negative_sims[0]])
# ax1.set_title("Negative Similarity Scores for all Batches After Final Epoch")
# ax1.set_xlabel("Similarity Score")
# ax1.set_ylabel('Frequency')

# ax2.hist([num * temp_value for num in final_positive_sims[0]])
# ax2.set_title("Positive Similarity Scores for all Batches After Final Epoch")
# ax2.set_xlabel("Similarity Score")
# ax2.set_ylabel('Frequency')

# plt.show()
# model = model.to('cpu')
original_model = model.module
original_model.to('cpu')
original_model.eval()

# Initialize lists to store features and labels
model_rep_list_trained = []

# Iterate over the DataLoaders
with torch.no_grad():  # Disable gradient computation for efficiency
    # for data_loader in dataloader_list:
    for batch_idx, (data) in enumerate(hard_dataloader_train):
        data = data[0].to(torch.float32)
        features = original_model(data)
        model_rep_list_trained.append(features)

# Convert lists to tensors
model_rep_trained = torch.cat(model_rep_list_trained, dim=0)

reducer = umap.UMAP(metric='cosine')

# Compute the mean and standard deviation for each row
mean = model_rep_trained.mean(dim=1, keepdim=True)
std = model_rep_trained.std(dim=1, keepdim=True, unbiased=False)

trained_rep_umap = reducer.fit_transform(model_rep_trained.clone().detach().numpy())

plt.figure()
plt.title("Data UMAP Representation Through the Trained Model")
plt.scatter(trained_rep_umap[:,0], trained_rep_umap[:,1], c = simple_tweetyclr.mean_colors_per_minispec[hard_indices_train,:])
plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
plt.savefig(f'{simple_tweetyclr.folder_name}/UMAP_of_trained_model.png')
plt.show()


# NOW FOR TESTING DATA

# Initialize lists to store features and labels
model_rep_list_trained_validation = []

# Iterate over the DataLoaders
with torch.no_grad():  # Disable gradient computation for efficiency
    # for data_loader in dataloader_list:
    for batch_idx, (data) in enumerate(hard_dataloader_test):
        data = data[0].to(torch.float32)
        features = original_model(data)
        model_rep_list_trained_validation.append(features)

# Convert lists to tensors
model_rep_trained_validation = torch.cat(model_rep_list_trained_validation, dim=0)

reducer = umap.UMAP(metric='cosine')

trained_rep_umap_validation = reducer.fit_transform(model_rep_trained_validation.clone().detach().numpy())

plt.figure()
plt.title("Data UMAP Representation Through the Trained Model: Validation Data")
plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
plt.scatter(trained_rep_umap_validation[:,0], trained_rep_umap_validation[:,1], c = simple_tweetyclr.mean_colors_per_minispec[hard_indices_test,:])
plt.savefig(f'{simple_tweetyclr.folder_name}/UMAP_of_trained_model_validation_data.png')
plt.show()


plt.figure()
plt.title("Loss Curve")
plt.plot(training_contrastive_loss[1:], color = 'blue', label = 'training')
plt.plot(validation_contrastive_loss[1:], color = 'orange', label = 'validation')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig(f'{simple_tweetyclr.folder_name}/loss_curves.png')
plt.show()

simple_tweetyclr.plot_UMAP_embedding(trained_rep_umap,  simple_tweetyclr.mean_colors_per_minispec[hard_indices_train,:], embeddable_images_train, f'{simple_tweetyclr.folder_name}/Plots/UMAP_of_trained_model.html', saveflag = True)
simple_tweetyclr.plot_UMAP_embedding(trained_rep_umap_validation,  simple_tweetyclr.mean_colors_per_minispec[hard_indices_test,:], embeddable_images_test, f'{simple_tweetyclr.folder_name}/Plots/UMAP_of_trained_model_testing.html', saveflag = True)

# =============================================================================
# # DOWNSTREAM EVALUATION
# =============================================================================

# Let's do K-Means + K-Fold CV with Silhouette score as the evaluation metric. This will allow me to determine how many clusers to use. 

X = trained_rep_umap.copy()

import hdbscan
clusterer = hdbscan.HDBSCAN(min_cluster_size = 100, gen_min_span_tree=True)
# Get the cluster labels

clusterer.fit(X)

# Get the cluster labels
labels = clusterer.labels_

# Number of clusters (excluding noise points)
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

print("Number of clusters:", n_clusters)

plt.figure()
plt.scatter(X[:,0], X[:,1], c=clusterer.labels_, cmap='viridis', s=50)
plt.suptitle("HDBSCAN Clustering of Trained UMAP+TweetyCLR Representation of Training Data")
plt.title(f'Number of Clusters Detected: {n_clusters}')
plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
plt.savefig(f'{simple_tweetyclr.folder_name}/hdbscan_plot_trained_model_training_data.png')
plt.show()


# HDBSCAN On confused region raw UMAP 

X = simple_tweetyclr.umap_embed_init[hard_indices_train,:].copy()

import hdbscan
clusterer = hdbscan.HDBSCAN(min_cluster_size=100, gen_min_span_tree=True)
# Get the cluster labels

clusterer.fit(X)

# Get the cluster labels
labels = clusterer.labels_

# Number of clusters (excluding noise points)
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

print("Number of clusters:", n_clusters)

plt.figure()
plt.scatter(X[:,0], X[:,1], c=clusterer.labels_, cmap='viridis', s=50)
plt.suptitle("HDBSCAN Clustering of UMAP Representation of Training Data")
plt.title(f'Number of Clusters Detected: {n_clusters}')
plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
plt.savefig(f'{simple_tweetyclr.folder_name}/hdbscan_plot_UMAP_training_data.png')
plt.show()





# Number of splits for K-Fold
n_splits = 10

# Range of the number of clusters
cluster_range = range(2, 10)

simple_tweetyclr.downstream_clustering(X, n_splits, cluster_range)

# plt.figure()
# plt.scatter(X[:,0], X[:,1], c=clusterer.labels_, cmap='Spectral', s=50)
# plt.title("HDBSCAN Clustering")
# plt.show()



# Validation data
X = trained_rep_umap_validation.copy()

clusterer = hdbscan.HDBSCAN(min_cluster_size=100, gen_min_span_tree=True)
# Get the cluster labels

clusterer.fit(X)

# Get the cluster labels
labels = clusterer.labels_

# Number of clusters (excluding noise points)
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

print("Number of clusters:", n_clusters)

plt.figure()
plt.scatter(X[:,0], X[:,1], c=clusterer.labels_, cmap='viridis', s=50)
plt.title("HDBSCAN Clustering of Trained UMAP+TweetyCLR Representation of Validation Data")
plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
plt.savefig(f'{simple_tweetyclr.folder_name}/hdbscan_plot_trained_model_validation_data.png')
plt.show()








# # Number of splits for K-Fold
# n_splits = 5

# # Range of the number of clusters
# cluster_range = range(2, 10)

# simple_tweetyclr.downstream_clustering(X, n_splits, cluster_range)


# Let's do supervised classification on non-transition slices only
X = model_rep_trained.clone().detach().cpu().numpy()

# Store indices of rows with 3 or more unique labels
transition_slice_indices = []

for i, row in enumerate(stacked_labels_train):
    if len(np.unique(row)) >= 3:
        transition_slice_indices.append(i)

transition_slices_indices = np.array(transition_slice_indices)

X = np.delete(X, transition_slice_indices, axis = 0)
# X = np.delete(trained_rep_umap, transition_slice_indices, axis = 0)
labels = np.delete(stacked_labels_train, transition_slice_indices, axis = 0)
mcpm = np.delete(simple_tweetyclr.mean_colors_per_minispec[hard_indices_train,:], transition_slice_indices, axis = 0)

# Function to find the first nonzero element in each row
def first_nonzero(arr, axis, invalid_val=-1):
    mask = arr != 0
    return np.where(mask.any(axis=axis), mask.argmax(axis=axis), invalid_val)

# Extract first nonzero element from each row
first_nonzero_indices = first_nonzero(labels, axis=1)
first_nonzero_elements = [labels[i, idx] if idx >= 0 else None for i, idx in enumerate(first_nonzero_indices)]

first_nonzero_elements = [0 if x is None else x for x in first_nonzero_elements]

labels = np.array(first_nonzero_elements)

# # test case: delete everything that is not the top 2 categories
# to_delete = np.where((labels != 13) & (labels != 2))[0]

# X = np.delete(X, to_delete, axis = 0)
# labels = np.delete(labels, to_delete, axis = 0)

X = torch.tensor(X)
labels = torch.tensor(labels)

# Create a mapping from original labels to a range 0 to N-1
unique_labels = torch.unique(labels)
label_to_idx = {label.item(): idx for idx, label in enumerate(unique_labels)}

# Map labels to consecutive indices
mapped_labels = torch.tensor([label_to_idx[label.item()] for label in labels])

# Create DataLoader
dataset = TensorDataset(X, mapped_labels)
train_loader = DataLoader(dataset, batch_size=16, shuffle=True)

class LinearClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LinearClassifier, self).__init__()
        self.fc = nn.Linear(input_size, num_classes)

    def forward(self, x):
        return self.fc(x)
    
    

# Example usage
input_size = 320  # Number of input features
num_classes = unique_labels.shape[0]  # Number of classes (change as needed)

clf = LinearClassifier(input_size, num_classes)
# Loss function and optimizer
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(clf.parameters(), lr=0.01)

loss_list = []
# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = clf(inputs)
        loss = loss_function(outputs, targets)
        loss.backward()
        optimizer.step()
        loss_list.append(loss.item())

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


# =============================================================================
# Calculate Accuracy
# =============================================================================


clf.eval()  # Set the model to evaluation mode
predictions = []
targets_list = []
correct = 0
total = 0
logit_scores = []
with torch.no_grad():  # No need to track gradients for validation
    for inputs, targets in train_loader:
        outputs = clf(inputs)
        _, predicted = torch.max(outputs.data, 1)  # Get the index of the max log-probability
        predictions.append(predicted.numpy())
        targets_list.append(targets.numpy())
        logit_scores.append(outputs.data)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()

accuracy = correct / total
print(f'Accuracy of the model on the test set: {accuracy * 100:.2f}%')

predictions = np.concatenate((predictions), axis = 0)
targets_list = np.concatenate((targets_list), axis = 0)

logit_scores = torch.cat(logit_scores)
probabilities = F.softmax(logit_scores, dim=1).detach().numpy()

# =============================================================================
# Calculate Confusion Matrix
# =============================================================================

from sklearn.metrics import confusion_matrix

# Compute the confusion matrix
cm = confusion_matrix(targets_list, predictions)

import seaborn as sns

# Plotting the confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d')
plt.title('Confusion Matrix')
plt.ylabel('Actual Labels')
plt.xlabel('Predicted Labels')

# Optional: If you want to display class labels on the axes
class_labels = [str(int(label.numpy())) for label in unique_labels]  # Replace with your class labels
plt.xticks(ticks=np.arange(len(class_labels)) + 0.5, labels=class_labels)
plt.yticks(ticks=np.arange(len(class_labels)) + 0.5, labels=class_labels, rotation=0)

plt.show()

# # =============================================================================
# # AUC Calculation
# # =============================================================================

# # Assuming 'outputs' is your model's raw output and 'true_labels' are your actual labels
# logits = logit_scores
# true_labels = mapped_labels  # Replace with your actual labels

# # Convert logits to probabilities
# probabilities = F.softmax(logits, dim=1).detach().numpy()

# from sklearn.metrics import roc_auc_score

# # Calculate OvA AUC
# auc_score = roc_auc_score(true_labels, probabilities, multi_class='ovr')

# print("One-vs-All AUC:", auc_score)



# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.metrics import roc_curve, auc
# from sklearn.preprocessing import label_binarize

# # Assuming 'y_true' is your true labels and 'y_scores' are the probability scores from your model
# # 'y_true' should be a 1D array of labels, and 'y_scores' should be a 2D array where each column corresponds to a class
# n_classes = len(np.unique(mapped_labels))

# # Binarize the labels for OvR calculation
# y_true_binary = label_binarize(mapped_labels, classes=np.arange(n_classes))

# # Calculate ROC curve and ROC area for each class
# fpr = dict()
# tpr = dict()
# roc_auc = dict()
# for i in range(n_classes):
#     fpr[i], tpr[i], _ = roc_curve(y_true_binary[:, i], probabilities[:, i])
#     roc_auc[i] = auc(fpr[i], tpr[i])

# # Plot all ROC curves
# plt.figure(figsize=(8, 6))
# colors = iter(plt.cm.rainbow(np.linspace(0, 1, n_classes)))
# for i, color in zip(range(n_classes), colors):
#     plt.plot(fpr[i], tpr[i], color=color, lw=2,
#              label='ROC curve of class {0} (area = {1:0.2f})'
#              ''.format(i, roc_auc[i]))

# plt.plot([0, 1], [0, 1], 'k--', lw=2)
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Some extension of Receiver operating characteristic to multi-class')
# plt.legend(loc="lower right")
# plt.show()


def top_k_accuracy(y_true, y_pred, k=5):
    """
    Calculate top-k accuracy.
    
    Parameters:
    y_true (Tensor): True labels.
    y_pred (Tensor): Predictions from the model. Assumes that predictions are logits or probabilities.
    k (int): Top k predictions to consider.
    
    Returns:
    float: Top-k accuracy.
    """
    # Get the top k predictions for each sample
    top_k_preds = torch.topk(y_pred, k, dim=1).indices

    # Check if the true labels are in the top k predictions
    correct = top_k_preds.eq(y_true.view(-1, 1).expand_as(top_k_preds))

    # Calculate accuracy
    top_k_accuracy = correct.sum().float() / y_true.size(0)
    return top_k_accuracy.item()

# Example usage
# Assuming y_true are your true labels and y_pred are your model's predictions
y_true = torch.tensor([2, 3, 1, 0, 4])  # Example true labels
y_pred = torch.randn(5, 10)  # Example predictions (random in this case)

k_value = 1
accuracy = top_k_accuracy(torch.tensor(targets_list), torch.tensor(probabilities), k=k_value)
print(f"Top-{k_value} accuracy: {accuracy * 100:.2f}%")



# For the committee presentation, let's see how normalized similarlity scores change as a function of temperature

cosine_sim = np.linspace(0.001,1,1000)

tau_value = [0.02, 0.5, 1.0]

normalized_sim = []

for i in np.arange(len(tau_value)):
    normalized_sim.append(np.exp(cosine_sim/tau_value[i]))
    
plt.figure()
plt.title("Scaled Cosine Similarity Scores")
plt.plot(cosine_sim, normalized_sim[0], label = f'Temperature = {tau_value[0]}', color = 'blue', linewidth = 2)
plt.plot(cosine_sim, normalized_sim[2], label = f'Temperature = {tau_value[2]}', color = 'green', linewidth = 2)
plt.xlabel("Cosine Similarity")
plt.ylabel("Scaled Similarity")
plt.legend()
plt.show()


# # Extract the learned positive and negative similarity scores

# model.eval()

# for batch_data in enumerate(zip(*dataloader_list_test)):
#     data_list = []
#     label_list = []
#     a = batch_data[1]
#     for idx in np.arange(len(a)):
#         data_list.append(a[idx][0])
    
# # for batch_idx, ((data1, labels1), (data2, labels2)) in enumerate(contrastive_loader):
#     data = torch.cat((data_list), dim = 0)
#     data = data.unsqueeze(1)
#     # data = data.reshape(a[idx][0].shape[0], len(a), a[idx][0].shape[1], a[idx][0].shape[2])
#     # labels = labels1
#     if torch.cuda.is_available():
#         data = data.cuda()
#     data = torch.autograd.Variable(data,False)
#     bsz = a[idx][0].shape[0]
#     data = data.to(torch.float32)
#     features = model(data)
#     norm = features.norm(p=2, dim=1, keepdim=True)
#     epsilon = 1e-12
#     # Add a small epsilon to prevent division by zero
#     normalized_tensor = features / (norm + epsilon)
#     split_features = torch.split(normalized_tensor, [bsz]*len(a), dim=0)
#     split_features = [split.unsqueeze(1) for split in split_features]

#     validation_features = torch.cat(split_features, dim = 1)

#     validation_loss, validation_negative_similarities, validation_positive_similarities = criterion(validation_features)


















