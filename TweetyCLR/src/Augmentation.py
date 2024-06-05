'''
This file will contain the different augmentations possible
'''
import torch

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

def white_noise_augmentation(windows, k=1, noise_std=1.0, noise_mean=0, seed=295):
    augmented_images = []
    for _ in range(k):
        # Generate white noise for each augmented image separately
        noise = torch.randn_like(windows) * noise_std + noise_mean
        # Add white noise to input tensor
        augmented_images.append(windows + noise)
    augmented_images = torch.stack(augmented_images).permute(1,0,2,3,4)

    return augmented_images

