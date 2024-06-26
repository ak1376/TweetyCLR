import torch
from torch.utils.data import Dataset, DataLoader
import random
from tqdm import tqdm

class ContrastiveDataset(Dataset):
    def __init__(self, data, labels, custom_transform=None, **kwargs):
        self.data = data
        self.labels = labels
        self.custom_transform = custom_transform
        self.kwargs = kwargs


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tensor = self.data[idx]
        label = self.labels[idx]
        if self.custom_transform:

            tensor1 = self.custom_transform(tensor)
            tensor2 = self.custom_transform(tensor)
        
            # random_mean_1 = random.randint(0, 10)
            # random_std_1 = random.randint(0, 10)

            # random_mean_2 = random.randint(0, 10)
            # random_std_2 = random.randint(0, 10)

            # tensor1 = self.custom_transform(tensor, k = 1, noise_std = random_std_1, noise_mean = random_mean_1)
            # tensor2 = self.custom_transform(tensor, k = 1, noise_std = random_std_2, noise_mean = random_mean_2)  # Apply the same transformation for the second tensor
        return tensor1, tensor2, label

# Define a custom data loader
class ContrastiveDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __iter__(self):
        for batch in super().__iter__():
            batch_size = len(batch)
            original_batch = []
            augmented_batch = []
            labels_batch = []
            original_batch.append(batch[0])
            augmented_batch.append(batch[1])
            labels_batch.append(batch[2])
            original_batch = torch.stack(original_batch).squeeze(0)  # Original samples
            augmented_batch = torch.stack(augmented_batch).squeeze(0)  # Augmented samples
            labels_batch = torch.stack(labels_batch).squeeze(0)  # Labels
            yield torch.cat((original_batch, augmented_batch), dim=0), labels_batch

# class ContrastiveDataset(Dataset):
#     def __init__(self, anchors, positives, anchor_labels):
#         """
#         Args:
#             anchors (torch.Tensor): Tensor of anchors.
#             positives (torch.Tensor): Tensor of positive augmentations.
#             anchor_labels (torch.Tensor): Tensor of labels for anchors.
#         """
#         self.anchors = anchors
#         self.positives = positives
#         self.anchor_labels = anchor_labels

#     def __len__(self):
#         return len(self.anchors)

#     def __getitem__(self, idx):
#         anchor = self.anchors[idx]
#         positive = self.positives[idx]
#         label = self.anchor_labels[idx]
#         return anchor, positive, label

# def contrastive_collate_fn(batch, dataset):
#     anchors, positives, labels = zip(*batch)
#     anchors = torch.stack(anchors)
#     positives = torch.stack(positives)
#     anchor_labels = torch.stack(labels)

#     # Create the minibatch of negative samples by random sampling
#     batch_size = anchors.size(0)
#     k = positives.shape[1]

#     num_negatives_per_anchor = batch_size - k - 1

#     dataset_length = len(dataset)

#     # Create a tensor to hold negative indices for each anchor
#     neg_indices = []

#     for i in tqdm(range(batch_size), desc = 'Sampling Negatives'):
#         # Exclude the anchor itself and k positive samples
#         excluded_indices = set([i] + list(range(i + 1, i + k + 1)))

#         # Determine the number of negative samples to sample for this anchor
#         num_negatives_to_sample = min(num_negatives_per_anchor, dataset_length - len(excluded_indices))

#         # Sample negative indices excluding the anchor and positive samples
#         neg_indices_for_anchor = torch.tensor(list(set(torch.multinomial(torch.ones(dataset_length - len(excluded_indices)), num_negatives_to_sample)))) 

#         neg_indices.append(neg_indices_for_anchor)

#     neg_indices = torch.stack(neg_indices)
#     negatives = dataset.anchors[neg_indices,:,:,:].squeeze().unsqueeze(2)
#     negative_labels = dataset.anchor_labels[neg_indices,:]


#     return anchors, positives, negatives, anchor_labels, negative_labels

















# from torch.utils.data import Dataset

# class Custom_Contrastive_Dataset(Dataset):
#     def __init__(self, tensor_data, tensor_labels, transform=None):
#         self.data = tensor_data
#         self.labels = tensor_labels
#         self.transform = transform

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, index):
#         x = self.data[index]
#         lab = self.labels[index]
#         x = [x,lab]
#         x1 = self.transform(x) if self.transform else x

#         return [x1, lab]


# class TwoCropTransform:
#     """Create two crops of the same image"""
#     def __init__(self, transform):
#         self.transform = transform

#     def __call__(self, x):
#         # Get the two augmentations from jawn
#         aug = self.transform(x)
#         return [aug[i, :, :, :] for i in range(aug.shape[0])]