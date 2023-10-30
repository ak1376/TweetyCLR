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
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui
from matplotlib import cm
from PyQt5.QtCore import Qt


class Tweetyclr:
    def __init__(self, num_spec, window_size, stride, folder_name, all_songs_data, masking_freq_tuple, spec_dim_tuple, exclude_transitions = False, category_colors = None):
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
        self.category_colors = category_colors
        self.folder_name = folder_name
        self.all_songs_data = all_songs_data
        self.masking_freq_tuple = masking_freq_tuple
        self.freq_dim = spec_dim_tuple[1]
        self.time_dim = spec_dim_tuple[0]
        self.exclude_transitions = exclude_transitions

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
            labels = dat['labels']
            labels = labels.T


            # Let's get rid of higher order frequencies
            mask = (frequencies<self.masking_freq_tuple[1])&(frequencies>self.masking_freq_tuple[0])
            masked_frequencies = frequencies[mask]

            subsetted_spec = spec[mask.reshape(mask.shape[0],),:]
            
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
            # open a file for writing in binary mode
            with open(f'{self.folder_name}/category_colors.pkl', 'wb') as f:
                # write the dictionary to the file using pickle.dump()
                pickle.dump(self.category_colors, f)

        spec_for_analysis = stacked_specs.T
        window_labels_arr = []
        embedding_arr = []
        # Find the exact sampling frequency (the time in miliseconds between one pixel [timepoint] and another pixel)
        print(times.shape)
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
            labels_for_window = stacked_labels[i:i+self.window_size, :]
            # Reshape the syllable labels for the window into a 1D array
            labels_for_window = labels_for_window.reshape(1, labels_for_window.shape[0]*labels_for_window.shape[1])
            # Populate the empty lists defined above
            stacked_windows.append(window)
            stacked_labels_for_window.append(labels_for_window)
            stacked_window_times.append(window_times)

        # Convert the populated lists into a stacked numpy array
        stacked_windows = np.stack(stacked_windows, axis = 0)
        stacked_windows = np.squeeze(stacked_windows)

        stacked_labels_for_window = np.stack(stacked_labels_for_window, axis = 0)
        stacked_labels_for_window = np.squeeze(stacked_labels_for_window)

        stacked_window_times = np.stack(stacked_window_times, axis = 0)
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
        image = image.convert('RGB')
        # show PIL image
        im_file = BytesIO()
        img_save = image.save(im_file, format='PNG')
        im_bytes = im_file.getvalue()
    
        img_str = "data:image/png;base64," + base64.b64encode(im_bytes).decode()
        return img_str
    
    
    def get_images(self, list_of_images):
        return list(map(self.embeddable_image, list_of_images))


    def compute_UMAP_decomp(self, zscored):
        # Perform a UMAP embedding on the dataset of mini-spectrograms
        reducer = umap.UMAP()
        embedding = reducer.fit_transform(zscored)

        return embedding

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

    def find_slice_actual_labels(self, stacked_labels_for_window):
        al = []
        for i in np.arange(stacked_labels_for_window.shape[0]):
            arr = stacked_labels_for_window[i,:]
            unique_elements, counts = np.unique(arr, return_counts=True)
            # print(unique_elements)
            # print(counts)
            sorted_indices = np.argsort(-counts)
            val = unique_elements[sorted_indices[0]]
            if val == 0:
                if unique_elements.shape[0]>1:
                    val = unique_elements[sorted_indices[1]]
            al.append(val)

        actual_labels = np.array(al)
        
        self.actual_labels = actual_labels

    def shuffling(self, shuffled_indices = None):
        
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
        stacked_labels_for_analysis_modeling= self.stacked_labels_for_window[shuffled_indices,:]
        mean_colors_per_minispec_for_analysis_modeling = self.mean_colors_per_minispec[shuffled_indices, :]
        
        stacked_windows_train = torch.tensor(dataset[anchor_indices,:])
        stacked_windows_train = stacked_windows_train.reshape(stacked_windows_train.shape[0], 1, self.time_dim, self.freq_dim)
        anchor_indices = anchor_indices
        # self.train_indices = np.array(training_indices)
        
        mean_colors_per_minispec_train = self.mean_colors_per_minispec[anchor_indices,:]
        stacked_labels_train = self.stacked_labels_for_window[anchor_indices,:]
        
        
        anchor_train_indices = anchor_indices
        
        return stacked_windows_train, stacked_labels_train, mean_colors_per_minispec_train, anchor_indices 
        
        
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
            

        return positive_aug_data

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
    

class DataPlotter:
    def __init__(self):


        # Setup main window/layout
        pg.setConfigOptions(imageAxisOrder='row-major')
        self.app = pg.mkQApp()


        # Instantiate window
        self.win = pg.GraphicsLayoutWidget()
        self.win.setWindowTitle('Embedding Analysis')

        # Behave plot
        self.behavePlot = self.win.addPlot()


        # Define bottom plot 
        self.win.nextRow()
        self.embPlot = self.win.addPlot()


        self.setupPlot()
        

    def setupPlot(self):

        # Setup behave plot for img
        self.imgBehave = pg.ImageItem()
        self.behavePlot.addItem(self.imgBehave)
        self.behavePlot.hideAxis('left')
        # self.behavePlot.hideAxis('bottom')


        # Setup emb plot
        self.embPlot.hideAxis('left')
        self.embPlot.hideAxis('bottom')

    def clear_plots(self):
        self.embPlot.clear()
        self.behavePlot.clear()


    # Change to general one day?
    def set_behavioral_image(self,image_array):

        self.behave_array = image_array
        self.imgBehave.setImage(self.behave_array)
     

    def update(self):
        rgn = self.region.getRegion()

        findIndices = np.where(np.logical_and(self.startEndTimes[0,:] > rgn[0], self.startEndTimes[1,:] < rgn[1]))[0]
    
        self.newScatter.setData(pos = self.emb[findIndices,:])



        self.embPlot.setXRange(np.min(self.emb[:,0]) - 1, np.max(self.emb[:,0] + 1), padding=0)
        self.embPlot.setYRange(np.min(self.emb[:,1]) - 1, np.max(self.emb[:,1] + 1), padding=0)






    def accept_embedding(self,embedding,startEndTimes):

        self.emb = embedding
        self.startEndTimes = startEndTimes


        self.cmap = cm.get_cmap('hsv')
        norm_times = np.arange(self.emb.shape[0])/self.emb.shape[0]
        colors = self.cmap(norm_times) * 255
        self.defaultColors = colors.copy()
        self.scatter = pg.ScatterPlotItem(pos=embedding, size=5, brush=colors)
        self.embPlot.addItem(self.scatter)
        
        self.newScatter = pg.ScatterPlotItem(pos=embedding[0:10,:], size=10, brush=pg.mkBrush(255, 255, 255, 200))
        self.embPlot.addItem(self.newScatter)


        # Scale imgBehave 
        height,width = self.behave_array.shape

        x_start, x_end, y_start, y_end = 0, self.startEndTimes[1,-1], 0, height
        pos = [x_start, y_start]
        scale = [float(x_end - x_start) / width, float(y_end - y_start) / height]

        self.imgBehave.setPos(*pos)
        tr = QtGui.QTransform()
        self.imgBehave.setTransform(tr.scale(scale[0], scale[1]))
        self.behavePlot.getViewBox().setLimits(yMin=y_start, yMax=y_end)
        self.behavePlot.getViewBox().setLimits(xMin=x_start, xMax=x_end)

        print(self.startEndTimes)
        self.region = pg.LinearRegionItem(values=(0, self.startEndTimes[0,-1] / 10))
        self.region.setZValue(10)

        
        self.region.sigRegionChanged.connect(self.update)

        self.behavePlot.addItem(self.region)


        # consider where 

        self.embPlot.setXRange(np.min(self.emb[:,0]) - 1, np.max(self.emb[:,0] + 1), padding=0)
        self.embPlot.setYRange(np.min(self.emb[:,1]) - 1, np.max(self.emb[:,1] + 1), padding=0)




    def plot_file(self,A):

        self.clear_plots()
        self.setupPlot()

        # A = np.load(filePath)

        self.startEndTimes = A['embStartEnd']
        self.set_behavioral_image(A['behavioralArr'])

        # feed it (N by 2) embedding and length N list of times associated with each point
        self.accept_embedding(A['embVals'],A['embStartEnd'])



    def show(self):
        self.win.show()
        self.app.exec_()


