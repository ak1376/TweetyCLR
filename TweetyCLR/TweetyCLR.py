import sys
sys.path.append('src')
sys.path.append('utils')
from experiments import Experiment_Manager
import os
import numpy as np
from utils import save_umap_img, cosine_similarity_batch
from tqdm import tqdm
import torch
from train import Trainer
from model import Encoder
from loss import info_nce

params = {
    'Experiment_Name': 'Canary_Disentangling',
    'train_dir': '/Users/AnanyaKapoor/Desktop/llb16_data_matrices/canary_train', # Put the local folder. This folder would contain the training windows
    'test_dir': '/Users/AnanyaKapoor/Desktop/llb16_data_matrices/canary_test', # Put the local folder. This folder would contain the validation windows
    'train_contrastive_dir': None, 
    'test_contrastive_dir': None,
    'category_colors': None, # If you want to load a python dictionary with the syllable labels and their corresponding colors.
    'window_size': 100, 
    'stride': 50, 
    'masking_freq_tuple': (500, 7000), 
    'temperature': 0.02, 
    'num_epochs': 1, 
    'n_positive_augs' : 3,
    'umap_distance_metric': 'cosine', 
    'random_seed': 295, 
    'batch_size': 256, 
    'learning_rate': 0.001, 
    'device': 'cpu'
}

# There should be a call to the experiment manager
experiment = Experiment_Manager(params = params)

training_data, testing_data = experiment.preprocess_data()

# Preload the embed object (if exists)

try:
    embed_path_train = os.path.join(os.getcwd(), 'Experiments', params['Experiment_Name'], 'embed_train.npy')
    embed_train = np.load(embed_path_train)
except FileNotFoundError:
    embed_train = experiment.first_pass_umap(training_data)

try: 
    # Preload the embed object (if exists)
    embed_path_test = os.path.join(os.getcwd(), 'Experiments', params['Experiment_Name'], 'embed_test.npy')
    embed_test = np.load(embed_path_test)

except FileNotFoundError:
    embed_test = experiment.first_pass_umap(testing_data)

# Now I want to visualize the embeddings and save the images.
save_umap_img(embed_train, training_data, params['Experiment_Name'], mode = 'train')
save_umap_img(embed_test, testing_data, params['Experiment_Name'], mode = 'test')

stacked_windows_train, train_aug, stacked_labels_train = experiment.applying_augmentations(training_data, n_positive_augs=params['n_positive_augs'], temporal_aug = False, white_noise_aug = True)
stacked_windows_test, test_aug, stacked_labels_test = experiment.applying_augmentations(testing_data, n_positive_augs=params['n_positive_augs'], temporal_aug = False, white_noise_aug = True)

train_dataloader = experiment.create_dataloader(stacked_windows_train, train_aug, stacked_labels_train)
test_dataloader = experiment.create_dataloader(stacked_windows_test, test_aug, stacked_labels_test)

# Now that I have preprocessed the dataset and confirmed that the negative samples
# are actually of good quality, I will now train the CNN model.

# First let's pass the data through the untrained model
# This code will all go in train.py

# mdl = Encoder().to(torch.float32)
# for anchors, positives, negatives, anchor_labels, negative_labels in tqdm(train_dataloader, desc="Processing Batches"):
#     anchors = anchors.to(torch.float32)
#     positives = positives.to(torch.float32)
#     negatives = negatives.to(torch.float32) 
#     anchor_rep = mdl(anchors)
#     positive_rep = mdl(positives)
#     negative_rep = mdl(negatives)

#     loss_value = info_nce(ref = anchor_rep, pos = positive_rep, neg = negative_rep)

#     break

mdl = Encoder().to(torch.float32)
optimizer = torch.optim.Adam(mdl.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-08)

trainer = Trainer(model = mdl,
                  train_dataloader=train_dataloader,
                  test_dataloader=test_dataloader,
                  loss_fn = info_nce,
                  optimizer = optimizer,
                  params = experiment.params,
                  device = experiment.params['device'],
                  lr = experiment.params['learning_rate'],
                  num_epochs = experiment.params['num_epochs'],
                  batch_size = experiment.params['batch_size']
                  )

trainer.train()