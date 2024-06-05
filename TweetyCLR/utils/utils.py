#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 10:06:23 2023

@author: akapoor
"""

import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
import umap
import torch
from bokeh.plotting import figure, show, output_file, save
from bokeh.models import HoverTool, ColumnDataSource
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import base64
import io
from io import BytesIO
from tqdm import tqdm

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import os

plt.rcParams.update({'font.size': 20})
plt.rcParams['figure.figsize'] = [15, 15]  # width and height should be in inches, e.g., [10, 6]

import torch
import torch.nn.functional as F

def moving_average(x, w):
    '''
    For visualizing the loss curves
    '''
    
    return np.convolve(x, np.ones(w), 'valid') / w

def cosine_similarity_batch(anchor_labels, negative_labels):
    """
    Calculate the cosine similarity for each (anchor_label, negative_label) pair.

    Args:
        anchor_labels (torch.Tensor): Tensor of shape (batch_size, num_features)
        negative_labels (torch.Tensor): Tensor of shape (batch_size, num_features)

    Returns:
        torch.Tensor: Tensor of cosine similarities of shape (batch_size,)
    """
    # Flatten the labels if they are not already flattened
    if len(anchor_labels.shape) > 2:
        anchor_labels = anchor_labels.view(anchor_labels.size(0), -1).float()
    if len(negative_labels.shape) > 2:
        negative_labels = negative_labels.view(negative_labels.size(0), -1).float()
    
    # Calculate cosine similarities
    cosine_similarities = F.cosine_similarity(anchor_labels, negative_labels)
    
    return cosine_similarities

def save_umap_img(embed, data_object, experiment_name, mode = 'train'):
    mean_colors_per_minispec = data_object['mean_colors_per_minispec']
    plt.figure()
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.title(f'UMAP on Raw Sonograms')
    plt.scatter(embed[:,0], embed[:,1], c = mean_colors_per_minispec)
    path_to_save = os.path.join(os.getcwd(), 'Experiments', experiment_name,f'embed_{mode}.png')
    plt.savefig(path_to_save)


class Tweetyclr:
    def __init__(self, window_size, stride, folder_name, train_dir, test_dir, masking_freq_tuple, category_colors = None):
        '''

        '''
        self.window_size = window_size
        self.stride = stride
        self.category_colors = category_colors
        self.folder_name = folder_name
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.train_files = [os.path.join(self.train_dir, f) for f in os.listdir(self.train_dir)]
        self.test_files =  [os.path.join(self.test_dir, f) for f in os.listdir(self.test_dir)]
        self.masking_freq_tuple = masking_freq_tuple

    def first_time_analysis(self, mode = 'train', zscore = True, num_spec = 100):
        '''
        The default should be num_spec = length of all files. But there should be an optional argument
        '''

        stacked_labels = [] 
        stacked_specs = []
        if mode == 'train':
            files_of_interest = self.train_files
        elif mode == 'test':
            files_of_interest = self.test_files

        for i in tqdm(np.arange(num_spec)):
            # Extract the data within the numpy file. We will use this to create the spectrogram
            dat = np.load(files_of_interest[i])
            spec = dat['s']
            times = dat['t']
            frequencies = dat['f']
            labels = dat['labels']
            labels = labels.T


            # Let's get rid of higher order frequencies
            mask = (frequencies<self.masking_freq_tuple[1])&(frequencies>self.masking_freq_tuple[0])
            masked_frequencies = frequencies[mask]

            subsetted_spec = spec[mask.reshape(mask.shape[0],),:]

            if zscore == True:
                mean = np.mean(subsetted_spec)
                std = np.std(subsetted_spec)
                subsetted_spec = (subsetted_spec - mean) / std
                        
            stacked_labels.append(labels)
            stacked_specs.append(subsetted_spec)

            
        stacked_specs = np.concatenate((stacked_specs), axis = 1)
        stacked_labels = np.concatenate((stacked_labels), axis = 0)
        stacked_labels.shape = (stacked_labels.shape[0],1)


        # Get a list of unique categories (syllable labels)
        unique_categories = np.unique(stacked_labels)
        if self.category_colors == None:
            self.category_colors = {category: np.random.rand(3,) for category in unique_categories}
            self.category_colors[0] = np.zeros((3)) # SIlence should be black
            # # open a file for writing in binary mode
            # with open(f'{self.folder_name}/category_colors.pkl', 'wb') as f:
            #     # write the dictionary to the file using pickle.dump()
            #     pickle.dump(self.category_colors, f)

        spec_for_analysis = stacked_specs.T
        window_labels_arr = []
        embedding_arr = []
        # Find the exact sampling frequency (the time in miliseconds between one pixel [timepoint] and another pixel)
        print(times.shape)
        dx = np.diff(times)[0,0]

        # Calculate the number of windows
        num_windows = (spec_for_analysis.shape[0] - self.window_size) // self.stride + 1

        # Initialize arrays to hold the results
        stacked_windows = np.zeros((num_windows, self.window_size * spec_for_analysis.shape[1]))
        stacked_labels_for_window = np.zeros((num_windows, self.window_size * stacked_labels.shape[1]))
        stacked_window_times = np.zeros((num_windows, self.window_size))

        # Use a vectorized approach to populate the arrays
        for idx in tqdm(range(num_windows)):
            i = idx * self.stride
            window = spec_for_analysis[i:i + self.window_size, :].flatten()
            window_times = dx * np.arange(i, i + self.window_size)
            labels_for_window = stacked_labels[i:i + self.window_size, :].flatten()

            stacked_windows[idx, :] = window
            stacked_labels_for_window[idx, :] = labels_for_window
            stacked_window_times[idx, :] = window_times

        # # Convert the populated lists into a stacked numpy array
        # stacked_windows = np.stack(stacked_windows, axis = 0)
        # stacked_windows = np.squeeze(stacked_windows)

        # stacked_labels_for_window = np.stack(stacked_labels_for_window, axis = 0)
        # stacked_labels_for_window = np.squeeze(stacked_labels_for_window)

        # stacked_window_times = np.stack(stacked_window_times, axis = 0)
        # dict_of_spec_slices_with_slice_number = {i: stacked_windows[i, :] for i in range(stacked_windows.shape[0])}
        
        
        
        # For each mini-spectrogram, find the average color across all unique syllables
        mean_colors_per_minispec = np.zeros((stacked_labels_for_window.shape[0], 3))
        for i in np.arange(stacked_labels_for_window.shape[0]):
            list_of_colors_for_row = [self.category_colors[x] for x in stacked_labels_for_window[i,:]]
            all_colors_in_minispec = np.array(list_of_colors_for_row)
            mean_color = np.mean(all_colors_in_minispec, axis = 0)
            mean_colors_per_minispec[i,:] = mean_color

        self.stacked_windows = stacked_windows
        self.stacked_labels_for_window = stacked_labels_for_window
        self.mean_colors_per_minispec = mean_colors_per_minispec
        self.stacked_window_times = stacked_window_times
        self.masked_frequencies = masked_frequencies

        dict_of_data = {

            'stacked_windows': self.stacked_windows, 
            'stacked_labels_for_window': self.stacked_labels_for_window, 
            'mean_colors_per_minispec': self.mean_colors_per_minispec, 
            'stacked_window_times': self.stacked_window_times,
            'category_colors': self.category_colors

        }

        return dict_of_data

        
# def embeddable_image(self, data):
#     data = (data.squeeze() * 255).astype(np.uint8)
#     # convert to uint8
#     data = np.uint8(data)
#     image = Image.fromarray(data)
#     image = image.rotate(90, expand=True) 
#     image = image.convert('RGB')
#     # show PIL image
#     im_file = BytesIO()
#     img_save = image.save(im_file, format='PNG')
#     im_bytes = im_file.getvalue()

#     img_str = "data:image/png;base64," + base64.b64encode(im_bytes).decode()
#     return img_str


# def get_images(self, list_of_images):
#     return list(map(self.embeddable_image, list_of_images))


# def plot_UMAP_embedding(self, embedding, mean_colors_per_minispec, image_paths, filepath_name, saveflag = False):

#     # Specify an HTML file to save the Bokeh image to.
#     # output_file(filename=f'{self.folder_name}Plots/{filename_val}.html')
#     output_file(filename = f'{filepath_name}')

#     # Convert the UMAP embedding to a Pandas Dataframe
#     spec_df = pd.DataFrame(embedding, columns=('x', 'y'))


#     # Create a ColumnDataSource from the data. This contains the UMAP embedding components and the mean colors per mini-spectrogram
#     source = ColumnDataSource(data=dict(x = embedding[:,0], y = embedding[:,1], colors=mean_colors_per_minispec))


#     # Create a figure and add a scatter plot
#     p = figure(width=800, height=600, tools=('pan, box_zoom, hover, reset'))
#     p.scatter(x='x', y='y', size = 7, color = 'colors', source=source)

#     hover = p.select(dict(type=HoverTool))
#     hover.tooltips = """
#         <div>
#             <h3>@x, @y</h3>
#             <div>
#                 <img
#                     src="@image" height="100" alt="@image" width="100"
#                     style="float: left; margin: 0px 15px 15px 0px;"
#                     border="2"
#                 ></img>
#             </div>
#         </div>
#     """

#     p.add_tools(HoverTool(tooltips="""
#     """))
    
#     # Set the image path for each data point
#     source.data['image'] = image_paths
#     # source.data['image'] = []
#     # for i in np.arange(spec_df.shape[0]):
#     #     source.data['image'].append(f'{self.folder_name}/Plots/Window_Plots/Window_{i}.png')


#     save(p)
#     show(p)
    
    
# def negative_sample_selection(self, data_for_analysis, indices_of_interest, easy_negative_indices_to_exclude):
    
#     # Hard negatives first: within the bound box region, let's sample k spectrogram slices that have the lowest cosine similarity score to each anchor slice
    
#     cosine_sim = cosine_similarity(self.umap_embed_init[indices_of_interest,:])

#     # Rewrite the loop as a list comprehension
#     # List comprehension to find the indices of the 10 smallest values for each row
#     smallest_indices_per_row = [np.concatenate((
#         np.array([indices_of_interest[i]]),
#         np.array([indices_of_interest[int(np.argpartition(cosine_sim[i, :], self.hard_negatives)[:self.hard_negatives][np.argsort(cosine_sim[i, :][np.argpartition(cosine_sim[i, :], self.hard_negatives)[:self.hard_negatives]])])]])
#     )) for i in np.arange(len(indices_of_interest))
#     ]
    
#     # Easy negatives: randomly sample p points from outside the bound box 
#     # region.
    
#     # For the testing dataset, I want to have easy negatives that are not
#     # in common with the easy negatives from training. Therefore I have 
#     # introduced "easy_negative_indices_to_exclude". 
    
#     total_indices = np.arange(data_for_analysis.shape[0])
#     easy_indices = np.setdiff1d(total_indices, self.hard_indices)

#     if easy_negative_indices_to_exclude is not None:
#         easy_indices = np.setdiff1d(easy_indices, np.intersect1d(easy_indices, easy_negative_indices_to_exclude)) # For the testing dataset, easy_negative_indices_to_exclude will be the easy negative indices for the training dataset

#     batch_indices_list = []
#     batch_array_list = []

#     all_sampled_indices = [np.random.choice(easy_indices, size=self.easy_negatives, replace=False) for i in range(len(self.hard_indices))]
    
#     # Now combine the easy negatives and hard negatives together 
#     concatenated_indices = [
#         np.concatenate([smallest_indices_per_row[i], all_sampled_indices[i]])
#         for i in range(len(smallest_indices_per_row))
#     ]
    
#     total_indices = np.stack(concatenated_indices)
#     total_indices = total_indices.reshape(total_indices.shape[0]*total_indices.shape[1])

#     dataset = data_for_analysis[total_indices,:]
#     dataset = torch.tensor(dataset.reshape(dataset.shape[0], 1, self.time_dim, self.freq_dim))
    
#     return total_indices, dataset
    
    
# def downstream_clustering(self, X, n_splits, cluster_range):
    
#     # K-Fold cross-validator
#     kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

#     # Dictionary to store the average silhouette scores for each number of clusters
#     average_silhouette_scores = {}

#     # Evaluate K-Means over the range of cluster numbers
#     for n_clusters in cluster_range:
#         silhouette_scores = []

#         for train_index, test_index in kf.split(X):
#             # Split data into training and test sets
#             X_train, X_test = X[train_index], X[test_index]

#             # Create and fit KMeans model
#             kmeans = KMeans(n_clusters=n_clusters, random_state=42)
#             kmeans.fit(X_train)

#             # Predict cluster labels for test data
#             cluster_labels = kmeans.predict(X_test)

#             # Calculate silhouette score and append to list
#             score = silhouette_score(X_test, cluster_labels)
#             silhouette_scores.append(score)

#         # Average silhouette score for this number of clusters
#         average_silhouette_scores[n_clusters] = np.mean(silhouette_scores)

#     # Print average silhouette scores for each number of clusters
#     max_score = 0 
#     optimal_cluster_number = 0
#     for n_clusters, score in average_silhouette_scores.items():
#         print(f'Average Silhouette Score for {n_clusters} clusters: {score}')
#         if score>max_score:
#             max_score = score
#             optimal_cluster_number = n_clusters
            
#     kmeans = KMeans(n_clusters=optimal_cluster_number, random_state=42)
#     kmeans.fit(X)
#     # Predict cluster labels for test data
#     cluster_labels = kmeans.predict(X)

#     # Plotting the clusters
#     plt.figure()
#     plt.scatter(X[:,0], X[:,1], c=cluster_labels, cmap='viridis')
#     plt.title('K-Means Clustering on UMAP of Model Representation')
#     plt.xlabel('Feature 1')
#     plt.ylabel('Feature 2')
#     plt.show()

# def train_test_split(self, dataset, train_split_perc, hard_indices):
#     ''' 
#     The following is the procedure I want to do for the train_test_split.
#     This function creates the training and testing anchor indices 
    
#     '''
    
#     split_point = int(train_split_perc*hard_indices.shape[0])

#     training_indices = hard_indices[:split_point]
#     testing_indices = hard_indices[split_point:]

#     # Shuffle array1 using the shuffled indices
#     stacked_windows_for_analysis_modeling = dataset[hard_indices,:]

#     # Shuffle array2 using the same shuffled indices
#     stacked_labels_for_analysis_modeling= self.stacked_labels_for_window[hard_indices,:]
#     mean_colors_per_minispec_for_analysis_modeling = self.mean_colors_per_minispec[hard_indices, :]

#     # Training dataset

#     stacked_windows_train = torch.tensor(dataset[training_indices,:])
#     stacked_windows_train = stacked_windows_train.reshape(stacked_windows_train.shape[0], 1, self.time_dim, self.freq_dim)
#     # self.train_indices = np.array(training_indices)

#     mean_colors_per_minispec_train = self.mean_colors_per_minispec[training_indices,:]
#     stacked_labels_train = self.stacked_labels_for_window[training_indices,:]

#     # Testing dataset

#     stacked_windows_test = torch.tensor(dataset[testing_indices,:])
#     stacked_windows_test = stacked_windows_test.reshape(stacked_windows_test.shape[0], 1, self.time_dim, self.freq_dim)
#     # self.train_indices = np.array(training_indices)

#     mean_colors_per_minispec_test = self.mean_colors_per_minispec[testing_indices,:]
#     stacked_labels_test = self.stacked_labels_for_window[testing_indices,:]
    
    
#     return stacked_windows_train, stacked_labels_train, mean_colors_per_minispec_train, training_indices, stacked_windows_test, stacked_labels_test, mean_colors_per_minispec_test, testing_indices

def pretraining(epoch, model, contrastive_loader_train, contrastive_loader_test, optimizer, criterion, epoch_number, method='SimCLR'):
    "Contrastive pre-training over an epoch. Adapted from XX"
    negative_similarities_for_epoch = []
    ntxent_positive_similarities_for_epoch = []
    mean_pos_cos_sim_for_epoch = []
    mean_ntxent_positive_similarities_for_epoch = []
    metric_monitor = MetricMonitor()
    
    # TRAINING PHASE
    
    model.train()

    for batch_data in enumerate(contrastive_loader_train):
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

        training_loss, training_negative_similarities, training_positive_similarities = criterion(features)

        metric_monitor.update("Training Loss", training_loss.item())
        metric_monitor.update("Learning Rate", optimizer.param_groups[0]['lr'])
        
        if epoch_number !=0:
            optimizer.zero_grad()
            training_loss.backward()
            optimizer.step()

        negative_similarities_for_epoch.append(float(np.mean(training_negative_similarities.clone().detach().cpu().numpy())))

        ntxent_positive_similarities_for_epoch.append(float(np.mean(training_positive_similarities.clone().detach().cpu().numpy())))
        
        # # Calculate the mean cosine similarity of the model feature representation for the positive pairs.
        # # Slice the tensor to separate the two sets of features you want to compare
        
        split_features = torch.split(normalized_tensor, [bsz]*len(a), dim=0)
        split_features = [split.unsqueeze(1) for split in split_features]
        training_features = torch.cat(split_features, dim = 1)
        
        # VALIDATION PHASE
        
        model.eval()

        for batch_data in enumerate(contrastive_loader_test):
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

            validation_features = torch.cat(split_features, dim = 1)

            validation_loss, validation_negative_similarities, validation_positive_similarities = criterion(validation_features)
            
            metric_monitor.update("Validation Loss", validation_loss.item())


    # print("[Epoch: {epoch:03d}] Contrastive Pre-train | {metric_monitor}".format(epoch=epoch, metric_monitor=metric_monitor))
    print(f'Epoch: {epoch:03d} Contrastive Pre-train Loss: {training_loss:.3f}, Validation Loss: {validation_loss:.3f}')
    # return metric_monitor.metrics['Loss']['avg'], metric_monitor.metrics['Learning Rate']['avg'], negative_similarities_for_epoch, features, mean_pos_cos_sim_for_epoch, mean_ntxent_positive_similarities_for_epoch
    return metric_monitor.metrics['Training Loss']['avg'], metric_monitor.metrics['Validation Loss']['avg'], metric_monitor.metrics['Learning Rate']['avg'], training_features, validation_features, negative_similarities_for_epoch, ntxent_positive_similarities_for_epoch
