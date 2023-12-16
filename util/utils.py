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

import numpy as np
# import pyqtgraph as pg
# from pyqtgraph.Qt import QtGui
# from matplotlib import cm
# from PyQt5.QtCore import Qt
from sklearn.metrics.pairwise import cosine_similarity

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import KFold


class Tweetyclr:
    def __init__(self, num_spec, window_size, stride, folder_name, all_songs_data, masking_freq_tuple, spec_dim_tuple, category_colors = None):
        '''The init function should define:
            1. directory for bird
            2. directory for python files
            3. analysis path
            4. folder name 


            Additional tasks
            1. create the folder name if it does not exist already

        '''
        # self.bird_dir = bird_dir
        # self.directory = directory
        self.num_spec = num_spec
        self.window_size = window_size
        self.stride = stride
        # self.analysis_path = analysis_path
        self.folder_name = folder_name
        self.all_songs_data = all_songs_data
        self.masking_freq_tuple = masking_freq_tuple
        self.freq_dim = spec_dim_tuple[1]
        self.time_dim = spec_dim_tuple[0]
        self.category_colors = category_colors

        # Create the folder if it doesn't already exist
        if not os.path.exists(folder_name+"/Plots/Window_Plots"):
            os.makedirs(folder_name+"/Plots/Window_Plots")
            print(f'Folder "{folder_name}" created successfully.')
        else:
            print(f'Folder "{folder_name}" already exists.')

    def first_time_analysis(self):

        # For each spectrogram we will extract
        # 1. Each timepoint's syllable label
        # 2. The spectrogram itself
        stacked_labels = [] 
        stacked_specs = []
        for i in np.arange(self.num_spec):
            # Extract the data within the numpy file. We will use this to create the spectrogram
            dat = np.load(self.all_songs_data[i])
            spec = dat['s']
            times = dat['t']
            frequencies = dat['f']
            labels = dat['label']
            # labels = labels.T


            # Let's get rid of higher order frequencies
            mask = (frequencies<self.masking_freq_tuple[1])&(frequencies>self.masking_freq_tuple[0])
            masked_frequencies = frequencies[mask]

            subsetted_spec = spec[mask.reshape(mask.shape[0],),:]
            
            stacked_labels.append(labels)
            stacked_specs.append(subsetted_spec)

            
        stacked_specs = np.concatenate((stacked_specs), axis = 1)
        stacked_labels = np.concatenate((stacked_labels), axis = 0)
        # stacked_labels.shape = (stacked_labels.shape[0],1)


        # Get a list of unique categories (syllable labels)
        unique_categories = np.unique(stacked_labels)
        print(unique_categories)
        print(self.category_colors)
        if self.category_colors == None:
            self.category_colors = {category: np.random.rand(3,) for category in unique_categories}
        #     self.category_colors[0] = np.zeros((3)) # SIlence should be black
        #     # open a file for writing in binary mode
            with open(f'{self.folder_name}/category_colors.pkl', 'wb') as f:
                # write the dictionary to the file using pickle.dump()
                pickle.dump(self.category_colors, f)

        print(self.category_colors)
        spec_for_analysis = stacked_specs.T
        window_labels_arr = []
        embedding_arr = []
        # Find the exact sampling frequency (the time in miliseconds between one pixel [timepoint] and another pixel)
        # print(times.shape)
        dx = np.diff(times)[0,0]

        # We will now extract each mini-spectrogram from the full spectrogram
        stacked_windows = []
        # Find the syllable labels for each mini-spectrogram
        stacked_labels_for_window = []
        # Find the mini-spectrograms onset and ending times 
        stacked_window_times = []

        # The below for-loop will find each mini-spectrogram (window) and populate the empty lists we defined above.
        for i in range(0, spec_for_analysis.shape[0] - self.window_size + 1, self.stride):
            # Find the window
            window = spec_for_analysis[i:i + self.window_size, :]
            # Get the window onset and ending times
            window_times = dx*np.arange(i, i + self.window_size)
            # We will flatten the window to be a 1D vector
            window = window.reshape(1, window.shape[0]*window.shape[1])
            # Extract the syllable labels for the window
            labels_for_window = stacked_labels[i:i+self.window_size]
            # Reshape the syllable labels for the window into a 1D array
            # labels_for_window = labels_for_window.reshape(1, labels_for_window.shape[0]*labels_for_window.shape[1])
            # Populate the empty lists defined above
            stacked_windows.append(window)
            stacked_labels_for_window.append(labels_for_window)
            stacked_window_times.append(window_times)

        # Convert the populated lists into a stacked numpy array
        stacked_windows = np.stack(stacked_windows, axis = 0)
        stacked_windows = np.squeeze(stacked_windows)

        stacked_labels_for_window = np.stack(stacked_labels_for_window, axis = 0)
        stacked_labels_for_window = np.squeeze(stacked_labels_for_window)
        print(f'Stacked labels for window shape: {stacked_labels_for_window.shape}')

        stacked_window_times = np.stack(stacked_window_times, axis = 0)
        # dict_of_spec_slices_with_slice_number = {i: stacked_windows[i, :] for i in range(stacked_windows.shape[0])}

        # # For each mini-spectrogram, find the average color across all unique syllables
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
        self.stacked_specs = stacked_specs
        self.stacked_labels = stacked_labels
        # self.dict_of_spec_slices_with_slice_number = dict_of_spec_slices_with_slice_number


    # def embeddable_image(self, data, folderpath_for_slices, window_times, iteration_number):
    #     # This function will save an image for each mini-spectrogram. This will be used for understanding the UMAP plot.
        
    #     window_data = data[iteration_number, :]
    #     window_times_subset = window_times[iteration_number, :]
    
    #     window_data.shape = (self.window_size, int(window_data.shape[0]/self.window_size))
    #     window_data = window_data.T 
    #     window_times = window_times_subset.reshape(1, window_times_subset.shape[0])
    #     plt.pcolormesh(window_times, self.masked_frequencies, window_data, cmap='jet')
    #     # let's save the plt colormesh as an image.
    #     plt.savefig(f'{folderpath_for_slices}/Window_{iteration_number}.png')
    #     plt.close()
    
    def embeddable_image(self, data):
        data = (data.squeeze() * 255).astype(np.uint8)
        # convert to uint8
        data = np.uint8(data)
        image = Image.fromarray(data)
        image = image.rotate(90, expand=True) 
        image = image.convert('RGB')
        # show PIL image
        im_file = BytesIO()
        img_save = image.save(im_file, format='PNG')
        im_bytes = im_file.getvalue()
    
        img_str = "data:image/png;base64," + base64.b64encode(im_bytes).decode()
        return img_str
    
    
    def get_images(self, list_of_images):
        return list(map(self.embeddable_image, list_of_images))


    # def compute_UMAP_decomp(self, zscored):
    #     # Perform a UMAP embedding on the dataset of mini-spectrograms
    #     reducer = umap.UMAP()
    #     embedding = reducer.fit_transform(zscored)

    #     return embedding

    def plot_UMAP_embedding(self, embedding, mean_colors_per_minispec, image_paths, filepath_name, saveflag = False):

        # Specify an HTML file to save the Bokeh image to.
        # output_file(filename=f'{self.folder_name}Plots/{filename_val}.html')
        output_file(filename = f'{filepath_name}')

        # Convert the UMAP embedding to a Pandas Dataframe
        spec_df = pd.DataFrame(embedding, columns=('x', 'y'))


        # Create a ColumnDataSource from the data. This contains the UMAP embedding components and the mean colors per mini-spectrogram
        source = ColumnDataSource(data=dict(x = embedding[:,0], y = embedding[:,1], colors=mean_colors_per_minispec))


        # Create a figure and add a scatter plot
        p = figure(width=800, height=600, tools=('pan, box_zoom, hover, reset'))
        p.scatter(x='x', y='y', size = 7, color = 'colors', source=source)

        hover = p.select(dict(type=HoverTool))
        hover.tooltips = """
            <div>
                <h3>@x, @y</h3>
                <div>
                    <img
                        src="@image" height="100" alt="@image" width="100"
                        style="float: left; margin: 0px 15px 15px 0px;"
                        border="2"
                    ></img>
                </div>
            </div>
        """

        p.add_tools(HoverTool(tooltips="""
        """))
        
        # Set the image path for each data point
        source.data['image'] = image_paths
        # source.data['image'] = []
        # for i in np.arange(spec_df.shape[0]):
        #     source.data['image'].append(f'{self.folder_name}/Plots/Window_Plots/Window_{i}.png')


        save(p)
        show(p)

    # def find_slice_actual_labels(self, stacked_labels_for_window):
    #     al = []
    #     for i in np.arange(stacked_labels_for_window.shape[0]):
    #         arr = stacked_labels_for_window[i,:]
    #         unique_elements, counts = np.unique(arr, return_counts=True)
    #         # print(unique_elements)
    #         # print(counts)
    #         sorted_indices = np.argsort(-counts)
    #         val = unique_elements[sorted_indices[0]]
    #         if val == 0:
    #             if unique_elements.shape[0]>1:
    #                 val = unique_elements[sorted_indices[1]]
    #         al.append(val)

    #     actual_labels = np.array(al)
        
    #     self.actual_labels = actual_labels

    def shuffling(self, seed, shuffled_indices = None):
        
        np.random.seed(seed)
        
        if shuffled_indices is None:
            shuffled_indices = np.random.permutation(self.stacked_windows.shape[0])
                    
        self.shuffled_indices = shuffled_indices
        
        
    def train_test_split(self, dataset, train_split_perc, shuffled_indices):
        ''' 
        The following is the procedure I want to do for the train_test_split.
        
        '''
        
        # I want to make training indices to be the first 80% of the shuffled data
        split_point = int(train_split_perc*dataset.shape[0])
        
        anchor_indices = shuffled_indices[:split_point]

        # Shuffle array1 using the shuffled indices
        stacked_windows_for_analysis_modeling = dataset[shuffled_indices,:]
        # Shuffle array2 using the same shuffled indices
        # stacked_labels_for_analysis_modeling= self.stacked_labels_for_window[shuffled_indices,:]
        mean_colors_per_minispec_for_analysis_modeling = self.mean_colors_per_minispec[shuffled_indices, :]
        
        stacked_windows_train = torch.tensor(dataset[anchor_indices,:])
        stacked_windows_train = stacked_windows_train.reshape(stacked_windows_train.shape[0], 1, self.time_dim, self.freq_dim)
        anchor_indices = anchor_indices
        # self.train_indices = np.array(training_indices)
        
        # mean_colors_per_minispec_train = self.mean_colors_per_minispec[anchor_indices,:]
        # stacked_labels_train = self.stacked_labels_for_window[anchor_indices,:]
        
        
        anchor_train_indices = anchor_indices
        
        return stacked_windows_train, anchor_indices 
    
        # return stacked_windows_train, stacked_labels_train, mean_colors_per_minispec_train, anchor_indices 
        
    
    
    def negative_sample_selection(self, data_for_analysis, indices_of_interest, easy_negative_indices_to_exclude):
        
        # Hard negatives first: within the bound box region, let's sample k spectrogram slices that have the lowest cosine similarity score to each anchor slice
        
        cosine_sim = cosine_similarity(self.umap_embed_init[indices_of_interest,:])

        # Rewrite the loop as a list comprehension
        # List comprehension to find the indices of the 10 smallest values for each row
        smallest_indices_per_row = [np.concatenate((
            np.array([indices_of_interest[i]]),
            np.array([indices_of_interest[int(np.argpartition(cosine_sim[i, :], self.hard_negatives)[:self.hard_negatives][np.argsort(cosine_sim[i, :][np.argpartition(cosine_sim[i, :], self.hard_negatives)[:self.hard_negatives]])])]])
        )) for i in np.arange(len(indices_of_interest))
        ]
        
        # Easy negatives: randomly sample p points from outside the bound box region.
        
        # For the testing dataset, I want to have easy negatives that are not
        # in common with the easy negatives from training. Therefore I have 
        # introduced "easy_negative_indices_to_exclude". 
        
        total_indices = np.arange(data_for_analysis.shape[0])
        easy_indices = np.setdiff1d(total_indices, self.hard_indices)

        if easy_negative_indices_to_exclude is not None:
            easy_indices = np.setdiff1d(easy_indices, np.intersect1d(easy_indices, easy_negative_indices_to_exclude)) # For the testing dataset, easy_negative_indices_to_exclude will be the easy negative indices for the training dataset

        batch_indices_list = []
        batch_array_list = []

        all_sampled_indices = [np.random.choice(easy_indices, size=self.easy_negatives, replace=False) for i in range(len(self.hard_indices))]
        
        # Now combine the easy negatives and hard negatives together 
        concatenated_indices = [
            np.concatenate([smallest_indices_per_row[i], all_sampled_indices[i]])
            for i in range(len(smallest_indices_per_row))
        ]
        
        total_indices = np.stack(concatenated_indices)
        total_indices = total_indices.reshape(total_indices.shape[0]*total_indices.shape[1])

        dataset = data_for_analysis[total_indices,:]
        dataset = torch.tensor(dataset.reshape(dataset.shape[0], 1, self.time_dim, self.freq_dim))
        
        return total_indices, dataset
    
    
    def downstream_clustering(self, X, n_splits, cluster_range):
        
        # K-Fold cross-validator
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

        # Dictionary to store the average silhouette scores for each number of clusters
        average_silhouette_scores = {}

        # Evaluate K-Means over the range of cluster numbers
        for n_clusters in cluster_range:
            silhouette_scores = []

            for train_index, test_index in kf.split(X):
                # Split data into training and test sets
                X_train, X_test = X[train_index], X[test_index]

                # Create and fit KMeans model
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                kmeans.fit(X_train)

                # Predict cluster labels for test data
                cluster_labels = kmeans.predict(X_test)

                # Calculate silhouette score and append to list
                score = silhouette_score(X_test, cluster_labels)
                silhouette_scores.append(score)

            # Average silhouette score for this number of clusters
            average_silhouette_scores[n_clusters] = np.mean(silhouette_scores)

        # Print average silhouette scores for each number of clusters
        max_score = 0 
        optimal_cluster_number = 0
        for n_clusters, score in average_silhouette_scores.items():
            print(f'Average Silhouette Score for {n_clusters} clusters: {score}')
            if score>max_score:
                max_score = score
                optimal_cluster_number = n_clusters
                
        kmeans = KMeans(n_clusters=optimal_cluster_number, random_state=42)
        kmeans.fit(X)
        # Predict cluster labels for test data
        cluster_labels = kmeans.predict(X)

        # Plotting the clusters
        plt.figure()
        plt.scatter(X[:,0], X[:,1], c=cluster_labels, cmap='viridis')
        plt.title('K-Means Clustering on UMAP of Model Representation')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.show()

        
class Temporal_Augmentation:
    
    def __init__(self, total_dict, simple_tweetyclr, tau_in_steps):
        self.total_dict = total_dict
        self.tweetyclr_obj = simple_tweetyclr
        self.tau = tau_in_steps
    
    def __call__(self, x):
        batch_data = x[0]
        indices = x[1]
        
        # Find the number of augmentations we want to use (up until tau step ahead)
        num_augs = np.arange(1, self.tau+1, 1).shape[0]
    
        # Preallocate tensors with the same shape as batch_data
        
        positive_aug_data = torch.empty(num_augs, 1, self.tweetyclr_obj.time_dim, self.tweetyclr_obj.freq_dim)
        
        # positive_aug_1 = torch.empty_like(batch_data)
        # positive_aug_2 = torch.empty_like(batch_data)
        
        total_training_indices = list(self.total_dict.keys())
        
        positive_aug_indices = indices + np.arange(1,self.tau+1, 1)      
            
        if any(elem in indices for elem in np.sort(total_training_indices)[-self.tau:]):
            positive_aug_indices = indices - np.arange(1,self.tau+1, 1)   
            
        try:
            # Your code that might raise the exception
            for i in np.arange(num_augs):
                positive_aug_data[i, :,:,:] = torch.tensor(self.total_dict[int(positive_aug_indices[i])].reshape(batch_data.shape[0], 1, self.tweetyclr_obj.time_dim, self.tweetyclr_obj.freq_dim))
        
        except ValueError as e:
            print(f"Encountered KeyError: {e}. Press Enter to continue...")
            input()

        # Define the noise scale (e.g., 5% of the data range)
        noise_scale = 0.3

        # Generate uniform noise and scale it
        noise = torch.rand_like(positive_aug_data) * noise_scale

        # Add the noise to the original tensor
        noisy_tensor = positive_aug_data + noise

        # Clip values to be between 0 and 1
        noisy_tensor_clipped = torch.clamp(noisy_tensor, 0, 1)

        
        return noisy_tensor_clipped

class Custom_Contrastive_Dataset(Dataset):
    def __init__(self, tensor_data, tensor_labels, transform=None):
        self.data = tensor_data
        self.labels = tensor_labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index]
        lab = self.labels[index]
        x = [x,lab]
        x1 = self.transform(x) if self.transform else x

        return [x1, lab]


class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        # Get the two augmentations from jawn
        aug = self.transform(x)
        return [aug[i, :, :, :] for i in range(aug.shape[0])]
