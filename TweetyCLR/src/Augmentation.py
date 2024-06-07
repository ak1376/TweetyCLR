'''
This file will contain the different augmentations possible
'''
import torch
import numpy as np

def temporal_augmentation(windows, n_steps_ahead = 1):
    num_windows, channel, window_size, features = windows.shape
    # Initialize the augmented tensor
    augmented_windows = torch.zeros((n_steps_ahead, num_windows, channel, window_size, features), device=windows.device)
    for n in range(1, n_steps_ahead + 1):
        # Create an augmented tensor by shifting windows along the sequence axis
        shifted_windows = torch.zeros_like(windows)
        if n < num_windows:
            shifted_windows[:-n] = windows[n:]
            shifted_windows[-n:] = windows[-1]  # Repeat the last window to keep the tensor shape consistent
        else:
            shifted_windows.fill_(0)  # Or handle differently if n_steps_ahead >= num_windows
        
        augmented_windows[n - 1] = shifted_windows
    
    return augmented_windows

def white_noise_augmentation(windows, k=1, noise_std=2.0, noise_mean=0, seed=295):
    # np.random.seed(seed)
    augmented_images = []
    for _ in range(k):
        # Generate white noise for each augmented image separately
        noise = np.random.normal(loc=noise_mean, scale=noise_std, size=windows.shape)
        # Add white noise to input tensor
        augmented_images.append(windows + noise)
    if k > 1:
        augmented_images = np.stack(augmented_images, axis=0)
    else:
        augmented_images = augmented_images[0]


    return augmented_images

def cutout_augmentation(tensor, num_holes = 1, hole_size = 50):
    """
    Apply Cutout to a given tensor.

    Args:
        tensor (torch.Tensor): Input tensor of shape (C, H, W).
        num_holes (int): Number of holes to cut out from the image.
        hole_size (int): Size of each square hole.

    Returns:
        torch.Tensor: Tensor with cutout applied.
    """
    # Get the dimensions of the tensor
    c, h, w = tensor.size()

    # Make a copy of the tensor to apply cutout
    tensor_cutout = tensor.clone()

    for _ in range(num_holes):
        # Randomly choose the top-left corner of the square
        y = np.random.randint(0, h - hole_size)
        x = np.random.randint(0, w - hole_size)

        # Set the square region to zero (black)
        tensor_cutout[:, y:y + hole_size, x:x + hole_size] = 0

    return tensor_cutout