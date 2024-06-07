'''
Object that will hold the experiment manager details
'''

from utils import Tweetyclr
import os
import umap
from collections import ChainMap
from tqdm import tqdm
import numpy as np
import torch
from Augmentation import temporal_augmentation, white_noise_augmentation, cutout_augmentation
from Contrastive_Dataloaders import ContrastiveDataset, ContrastiveDataLoader
from torchvision import transforms


class Experiment_Manager:
    '''
    I want the experiment manager to take the dictionary of parameters a
    '''
    def __init__(self, params):
        self.params = params

        # I now want to create a folder for the experiment if it does not exist
        # Define path components
        folder = os.getcwd()
        subfolder = "Experiments"
        filename = self.params['Experiment_Name']

        self.folder_name = os.path.join(folder, subfolder, filename)
        os.makedirs(self.folder_name, exist_ok=True)

    
    def calculate_mean_colors_per_minispec(self, stacked_labels, category_colors):
        # Convert category_colors to a NumPy array for faster indexing
        unique_syllables = list(category_colors.keys())
        color_array = np.array([category_colors[syllable] for syllable in unique_syllables])

        # Create a mapping from syllable to index for fast lookup
        syllable_to_index = {syllable: idx for idx, syllable in enumerate(unique_syllables)}

        # Convert stacked_labels to indices using the syllable_to_index mapping
        indexed_labels = np.vectorize(syllable_to_index.get)(stacked_labels)

        # Initialize mean_colors_per_minispec
        mean_colors_per_minispec = np.zeros((stacked_labels.shape[0], 3))

        # Calculate mean colors for each mini-spectrogram
        for i in tqdm(range(stacked_labels.shape[0])):
            all_colors_in_minispec = color_array[indexed_labels[i]]
            mean_colors_per_minispec[i, :] = np.mean(all_colors_in_minispec, axis=0)

        return mean_colors_per_minispec

    
    
    def preprocess_data(self):
        '''
        This method will run the windowing and all the other preprocessing procedures
        '''

        # This object would need to take the training directory and the testing directory

        simple_tweetyclr = Tweetyclr(self.params['window_size'],
                                     self.params['stride'],
                                     self.folder_name,
                                     self.params['train_dir'], 
                                     self.params['test_dir'],
                                     self.params['masking_freq_tuple'],
                                     self.params['category_colors'])

       # Training mode 
        training_data = simple_tweetyclr.first_time_analysis(mode = 'train', num_spec = 150)
        
        # Testing mode
        testing_data = simple_tweetyclr.first_time_analysis(mode = 'test', num_spec = 100)


        # Training data and testing data have their own "category_colors" python dictionary
        # that have RGB arrays that map onto each key (syllable). I want to create a combined
        # "category_colors" object that takes the common keys and values from each dictionary 
        # and, if there are any keys that are different between training and testing
        # category_colors then I should also include those disparate keys with their respective
        # values. 

        # Assuming training_data and testing_data have a 'category_colors' attribute which is a dictionary
        train_category_colors = training_data['category_colors']
        test_category_colors = testing_data['category_colors']

        # Combine the dictionaries
        combined_category_colors = dict(ChainMap(train_category_colors, test_category_colors))

        # Alternatively, if you want to ensure the values from the test dictionary take precedence:
        combined_category_colors = {**train_category_colors, **test_category_colors}

        training_data['category_colors'] = combined_category_colors
        testing_data['category_colors'] = combined_category_colors

        # Now let's recalculate the mean_colors_per_minispec for both training and testing.
        # mean_colors_per_minispec is a key in both training_data and testing_data

        # Assuming training_data and testing_data have the necessary attributes
        train_stacked_labels = training_data['stacked_labels_for_window']
        test_stacked_labels = testing_data['stacked_labels_for_window']

        # Assuming the combined category colors dictionary is used for consistency
        combined_category_colors = {**training_data['category_colors'], **testing_data['category_colors']}

        # Calculate mean colors for training data
        mean_colors_per_minispec_train = self.calculate_mean_colors_per_minispec(train_stacked_labels, combined_category_colors)

        # Calculate mean colors for testing data
        mean_colors_per_minispec_test = self.calculate_mean_colors_per_minispec(test_stacked_labels, combined_category_colors)

        # Store the results back to the training and testing data if needed
        training_data['mean_colors_per_minispec'] = mean_colors_per_minispec_train
        testing_data['mean_colors_per_minispec'] = mean_colors_per_minispec_test

        # stacked_windows_train = training_data['stacked_windows']
        # stacked_labels_for_window_train = training_data['stacked_labels_for_window']
        # mean_colors_per_minispec_train = training_data['mean_colors_per_minispec']

        # stacked_windows_test = testing_data['stacked_windows']
        # stacked_labels_for_window_test = testing_data['stacked_labels_for_window']
        # mean_colors_per_minispec_test = testing_data['mean_colors_per_minispec']

        return training_data, testing_data
        

    def first_pass_umap(self, data_object, embed = None):

        if embed == None:
            stacked_windows = data_object['stacked_windows']
            reducer = umap.UMAP(metric = self.params['umap_distance_metric'], random_state = self.params['random_seed'])
            embed = reducer.fit_transform(stacked_windows)
    
        return embed
    
    # def applying_augmentations(self, data_object, n_positive_augs = 1, temporal_aug = True, white_noise_aug = False):
    #     '''
    #     I will need to modify this function to allow for multiple augmentations. I won't do it right now though.
    #     '''
    #     if len(data_object['stacked_windows'].shape) < 4: 
    #         # Reshaping -- can probably make more elegant later
    #         stacked_windows = torch.tensor(data_object['stacked_windows'].reshape(data_object['stacked_windows'].shape[0], 1, self.params['window_size'],int(data_object['stacked_windows'].shape[1]//self.params['window_size'])))
    #         stacked_labels = torch.tensor(data_object['stacked_labels_for_window'])

    #     if temporal_aug == True:
    #         if white_noise_aug == True:
    #             temp_aug = temporal_augmentation(stacked_windows, n_steps_ahead = self.params['n_positive_augs'])[0,:,:,:]
    #             aug_data = white_noise_augmentation(temp_aug, k = self.params['n_positive_augs'], noise_std = 3.0, noise_mean = 0)
    #         elif white_noise_aug == False:
    #             aug_data = temporal_augmentation(stacked_windows, k = self.params['n_positive_augs'], n_steps_ahead = self.params['n_positive_augs'])[0,:,:,:]
    #     elif temporal_aug == False: 
    #         if white_noise_aug == True: 
    #             aug_data = white_noise_augmentation(stacked_windows, k = self.params['n_positive_augs'], noise_std = 3.0, noise_mean = 0)

    #     return stacked_windows, aug_data, stacked_labels
    
    def create_dataloader(self, stacked_windows, stacked_labels, temporal_aug = False, white_noise_aug = False, cutout_aug = True):
        '''
        Rewrite this a bit to include the temporal augmentations
        '''

        transform_list = []

        if temporal_aug:
            transform_list.append(temporal_augmentation)

        if white_noise_aug:
            transform_list.append(white_noise_augmentation)

        if cutout_aug:
            transform_list.append(cutout_augmentation)

        # Combine all transformations
        custom_transform = transforms.Compose(transform_list)

        dataset = ContrastiveDataset(stacked_windows, stacked_labels, custom_transform=custom_transform)
        dataloader = ContrastiveDataLoader(dataset, batch_size=self.params['batch_size'], shuffle=True)

        return dataloader

    



















    