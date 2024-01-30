#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 30 11:47:00 2023

@author: AnanyaKapoor
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 10:54:56 2023

This code will simulate a canary song but with no random walk and simulating 
each syllable separately. Each unique syllable will be defined by a 
10-dimensional multivariate Gaussian. Therefore, we will have a 10d parameter
value for each syllable occurrence, NOT each syllable-phrase occurrence


@author: AnanyaKapoor
"""

import numpy as np
import matplotlib.pyplot as plt 
import scipy.signal as signal
# import sounddevice as sd  
from scipy.io.wavfile import write
import pandas as pd
import seaborn as sns 
# import umap
import os 

folderpath = '/Users/AnanyaKapoor/Dropbox (University of Oregon)/Kapoor_Ananya/01_Projects/01_b_Canary_SSL/TweetyCLR_End_to_End/'
sns.set(style='white', context='notebook', rc={'figure.figsize':(14,10)})

plt.ioff()

sampling_freq = 44100

num_syllables = 5
num_short = 3
num_long = 2

mean_phi_0 = (np.random.uniform(0, 2*np.pi, num_syllables)).reshape(1, num_syllables)
mean_delta_phi = (np.random.uniform(-3*np.pi/2, 3*np.pi/2, num_syllables)).reshape(1, num_syllables) # In radians
mean_B = (np.random.uniform(300, 500, num_syllables)).reshape(1, num_syllables) # In Hz
mean_c = (np.random.uniform(40, 70, num_syllables)).reshape(1, num_syllables)
mean_f_0 = (np.random.uniform(800, 1500, num_syllables)).reshape(1, num_syllables) # In Hz

short_durations = np.random.uniform(30/1000, 90/1000, num_short)
long_durations = np.random.uniform(200/1000, 500/1000, num_long)
# short_repeats = np.random.randint(50, 100, num_short)
# long_repeats = np.random.randint(3, 5, num_long)

mean_T = np.concatenate((short_durations, long_durations))
# num_repeats = np.concatenate((short_repeats, long_repeats))

permutation = np.random.permutation(len(mean_T))

mean_T = mean_T[permutation]
mean_T.shape = (1, num_syllables)
# num_repeats = num_repeats[permutation]

mean_Z_1 = (np.random.uniform(0.88, 0.93, num_syllables)).reshape(1, num_syllables)
mean_Z_2 = (np.random.uniform(0.88, 0.93, num_syllables)).reshape(1, num_syllables)
mean_theta_1 = (np.random.uniform(0.01, np.pi/2, num_syllables)).reshape(1, num_syllables)
mean_theta_2 = (np.random.uniform(0.01, np.pi/2, num_syllables)).reshape(1, num_syllables)

# num_repeats = 50*np.ones((1, num_syllables)) # Simple example

mean_matrix = np.concatenate((mean_phi_0, mean_delta_phi, mean_B, mean_c, mean_f_0, mean_T, mean_Z_1, mean_Z_2, mean_theta_1, mean_theta_2), axis = 0)

# Let's find a random order of syllable phrases to simulate 
unique_syllables = np.arange(1, num_syllables+1)
# unique_syllables = np.arange(num_syllables)
syllable_phrase_order = unique_syllables.copy()
phrase_repeats = 5

num_songs = 1

radius_value = 0.01

songpath = f'{folderpath}num_songs_{num_songs}_num_syllables_{num_syllables}_phrase_repeats_{phrase_repeats}_radius_{radius_value}/'

if not os.path.exists(songpath):
    # Create the directory
    os.makedirs(songpath)
    print(f"Directory '{songpath}' created successfully.")
else:
    print(f"Directory '{songpath}' already exists.")


mean_matrix_df = pd.DataFrame(mean_matrix.T)
mean_matrix_df.columns = ['mean_phi_0','mean_delta_phi','mean_B','mean_c','mean_f_0','mean_T','mean_Z_1', 'mean_Z_2', 'mean_theta_1', 'mean_theta_2']

mean_matrix_df.to_csv(f'{songpath}mean_params_per_syllable.csv', index = True )

# For each song we want to store the following information: 
    # 1. The phrase order of each song (which will be different for every song)
    # 2. The acoustic parameters for each syllable repeat within the song 
    # 3. The spectrogram with labels 
    # 4. The audio representation
    # 5. The number of repeats per syllable phrase
    

syllable_phrase_order_songs = np.zeros((num_songs, syllable_phrase_order.shape[0]))
num_repeats_songs = np.zeros((num_songs, num_syllables*phrase_repeats))

for song_index in np.arange(num_songs):
    folderpath_song = f'{songpath}Song_{song_index}/'
    if not os.path.exists(folderpath_song):
        # Create the directory
        os.makedirs(folderpath_song)
        print(f"Directory '{songpath}' created successfully.")
    else:
        print(f"Directory '{songpath}' already exists.")
        

    np.random.shuffle(syllable_phrase_order) # ex: 0, 2, 1 means that we will simulate syllable 0 first, followed by 2 and then followed by 1
    syllable_phrase_order_songs[song_index,:] = syllable_phrase_order
    
    syllable_phrase_order_w_repeats = np.repeat(syllable_phrase_order, phrase_repeats) # Now add the number of phrase repeats 
    np.random.shuffle(syllable_phrase_order_w_repeats)
    

    phi_0_vector = []
    delta_phi_vector = []
    B_vector = []
    c_vector = []
    f_0_vector = []
    T_vector = []
    Z_1_vector = []
    Z_2_vector = []
    theta_1_vector = []
    theta_2_vector = []

    # Initializing empty arrays that will hold our signal wave, filtered wave, and enveloped wave
    
    total_signal_wave = np.array([])
    total_filtered = np.array([])
    total_envelope = np.array([])
    total_normalized_signal = np.array([])

    labels_per_sample = np.array([])
    
    # Sample parameters
    window_duration_seconds = 0.02  # 40 ms window
    window_size = int(sampling_freq * window_duration_seconds)
    overlap_fraction = 0.9       # 90 percent overlap           
    overlap = int(window_size * overlap_fraction) 
    
    low_frequency_check = 0 
    high_frequency_check = 0
    
    # f_0 = 0
    num_repeats_list = []
    # Double for loop: one over the syllable phrase and the other over the number of repeats of syllable
    phrase_duration_list = []
    for syl in syllable_phrase_order_w_repeats:
        # mu = mean_matrix[:,syl]
        # mean_duration = mu[5]
        # num_repeats = np.ceil(2 /mean_duration)
        # num_repeats_list.append(num_repeats)
            
        num_repeats = 0
        phrase_duration = 0
        
        while phrase_duration < 1.4:
    
            # We are going to ensure that each simulated parameter is within 1% of the mean value for the parameter. This will result in syllables with very little within-syllable variability
        
            # Draw acoustic parameters with respect to the mean vector corresponding to the syllable we are simulating
            mu = mean_matrix[:,syl-1]
            
            # Define the desired radius (strictly within 0.05 from the centroid)
            radius = radius_value
            
            # Number of random points to generate
            num_points = 1
            
            # Number of dimensions (size of the centroid array)
            num_dimensions = mu.shape[0]
            
            # Generate random directions (unit vectors) in num_dimensions-dimensional space
            random_directions = np.random.randn(num_points, num_dimensions)
            random_directions /= np.linalg.norm(random_directions, axis=1)[:, np.newaxis]
            
            # Generate random distances within the desired radius for each dimension
            random_distances = radius * np.random.rand(num_points) ** (1/num_dimensions)
            
            # Calculate the final random points within the hypersphere
            acoustic_params = mu + random_distances[:, np.newaxis] * random_directions
            acoustic_params.shape = (10,)
    
            
            # for param in np.arange(10):
            #     sim_param = (np.random.uniform(mu[param] - mu[param]*0.05, mu[param]+mu[param]*0.05, 1))
            #     acoustic_params = np.concatenate((acoustic_params, sim_param))
        
            # acoustic_params = np.random.multivariate_normal(mean_matrix[:,syl], covariance_matrix[syl, :, :])
            
            # tab = np.concatenate((mean_matrix[:,syl].reshape(10,1), acoustic_params.reshape(10,1)), axis = 1)
            # if low_frequency_check == 1:
            #     f_0 += 50
            #     acoustic_params[4] = f_0 
            # elif high_frequency_check == 1:
            #     f_0 -= 50
            #     acoustic_params[4] = f_0 
            
                
            
            phi_0 = acoustic_params[0]
            phi_0_vector.append(phi_0)
            
            delta_phi = acoustic_params[1]
            delta_phi_vector.append(delta_phi)
            
            B = acoustic_params[2]
            # B_vector.append(B)
            
            c = acoustic_params[3]
            c_vector.append(c)
            
            f_0 = acoustic_params[4]
            # f_0_vector.append(f_0)
            
            T = acoustic_params[5]
            T_vector.append(T)
            # print(T)
            
            Z_1 = acoustic_params[6]
            Z_1_vector.append(Z_1)
            
            Z_2 = acoustic_params[7]
            Z_2_vector.append(Z_2)
            
            theta_1 = acoustic_params[8]
            theta_1_vector.append(theta_1)
            
            theta_2 = acoustic_params[9]
            theta_2_vector.append(theta_2)
            
            # Let's create a table where we have the sampled acoustic parameters plotted against the mean acoustic parameters
            tab = np.concatenate((mean_matrix[:,syl-1].reshape(10,1), acoustic_params.reshape(10,1)), axis = 1)
            
            num_samples = int((T)*sampling_freq)
            t = np.linspace(0, ((T)), num_samples) 
    
            # Calculate the fundamental frequency across time
            f = f_0 + B*np.cos(phi_0 + delta_phi*t/T)
            
            # syllable_labels = np.repeat(syl, t.shape[0])
            # labels_per_sample = np.concatenate((labels_per_sample, syllable_labels))
            
            
            while np.min(f)<700:
                low_frequency_check = 1
                f_0+=50
                B -=20
                f = f_0 + B*np.cos(phi_0 + delta_phi*t/T)
            
            while np.max(f)>3000:
                high_frequency_check == 1
                f_0-=50
                B-=20
                f = f_0 + B*np.cos(phi_0 + delta_phi*t/T)
            
            # if np.min(f)<700:
            #     low_frequency_check = 1
            #     f_0+=50
            #     B-=20
            # else:
            #     low_frequency_check = 0
                
            # if np.max(f)>3000:
            #     high_frequency_check = 1
            #     f_0-=50
            #     B-=20
            # else:
            #     high_frequency_check = 0
                
            # if (low_frequency_check ==1) or (high_frequency_check == 1):
            #     # Recalculate the fundamental frequency across time with the new f_0 and B values
            #     f = f_0 + B*np.cos(phi_0 + delta_phi*t/T)
                
                    
            f_0_vector.append(f_0)
            B_vector.append(B)
            # It's the B*np.cos(phi_0_values + delta_phi_values*t/T) that gives the fundamental frequency its wavy shape. f_0 just shifts it up
            
            #     # Now let's calculate the harmonics 
            num_harmonics = 12
            theta_arr = np.zeros((num_harmonics, t.shape[0]))
            for k in np.arange(num_harmonics):
                # val = 2*np.pi*(k+1)*f.reshape(f.shape[0],)
                val = 2*np.pi*(k+1)*f_0*t + (2*np.pi*(k+1)*B*T/(delta_phi))*(np.sin((phi_0)+(delta_phi)/T*t) - np.sin((phi_0)))
                theta_arr[k, :] = val
                
            ## coefficients
            
            A_list = [1]
            for k in np.arange(2, (num_harmonics + 1)):
                coef = 1/(1+c*2**(k-1))
                # coef = 1
                A_list.append(coef)
                
            #     # Raw signal
                
            s_t_arr = np.zeros_like(t)
            
            for k in np.arange(len(A_list)):
                signal_val = A_list[k]*np.sin(theta_arr[k,:])
                s_t_arr += signal_val
            
            
            total_signal_wave = np.concatenate((total_signal_wave, s_t_arr))
                
            #     # Filtered signal
    
            r1_roots = Z_1 * np.exp(1j*theta_1)
            r2_roots = Z_2 * np.exp(1j*theta_2)
            roots = [r1_roots, np.conjugate(r1_roots), r2_roots, np.conjugate(r2_roots)]
            
            # Convert the roots to zeros, poles, and gain representation
            zeros = []
            poles = roots
            gain = 1.0
    
            # Convert zeros, poles, and gain to filter coefficients
            b, a = signal.zpk2tf(zeros, poles, gain)
    
            # Apply the all-pole filter to the input signal
            y_arr = signal.lfilter(b, a, s_t_arr)
    
            total_filtered = np.concatenate((total_filtered, y_arr))
            
            normalized_signal = np.zeros_like(y_arr)
    
            for i in range(0, len(y_arr) - window_size + 1, window_size - overlap):
                window = y_arr[i:i + window_size]  # Extract a window of the signal
                scaling_factor = 1.0 / np.max(np.abs(window))  # Calculate the scaling factor
                normalized_signal[i:i + window_size] = window * scaling_factor  # Normalize the window
    
            total_normalized_signal = np.concatenate((total_normalized_signal, normalized_signal))
                
            #     # Enveloped signal 
            
            # W_t = (0.42 + 0.5*np.cos(np.pi * t/T) + 0.08*np.cos(2*np.pi * t/T))
            W_t = 0.5 * (1 - np.cos(2 * np.pi * t / T))
                
            waveform_filtered_envelope = normalized_signal * W_t
            syllable_labels = np.repeat(syl, waveform_filtered_envelope.shape[0])
            syllable_labels[-441:] = 0
            # syllable_labels[np.argwhere(waveform_filtered_envelope == 0)] = 0
            
            labels_per_sample = np.concatenate((labels_per_sample, syllable_labels))
            
            
            total_envelope = np.concatenate((total_envelope, waveform_filtered_envelope))
            
            phrase_duration += waveform_filtered_envelope.shape[0]/44100
            num_repeats +=1
            
        phrase_duration_list.append(phrase_duration)
        num_repeats_list.append(num_repeats)
        
    frequencies, times, spectrogram = signal.spectrogram(total_envelope, fs=sampling_freq,
                                                        window='hamming', nperseg=256,
                                                        noverlap=128, nfft=512)
    
    plt.figure()
    plt.pcolormesh(times, frequencies, spectrogram, cmap='jet')
    plt.title(f'Spectrogram_of_Song_{song_index}')
    plt.savefig(f'{folderpath_song}Spectrogram_of_Song.png')
    
    # Calculate the number of samples and pixels
    num_samples = len(total_envelope)
    num_pixels = spectrogram.shape[1]
    
    # Create an array to store labels per pixel
    labels_per_pixel = np.zeros(num_pixels)
    
    # # Calculate the mapping between samples and pixels
    overlap = 128
    window_size = 256
    samples_per_pixel = (window_size - overlap)
    mapping = np.arange(0, num_samples - window_size + 1, samples_per_pixel)
    
    # Map each label to the corresponding time pixel in the spectrogram using majority voting
    for i in range(num_pixels):
        start_sample = mapping[i]
        end_sample = start_sample + samples_per_pixel
        labels_in_window = labels_per_sample[start_sample:end_sample]
        labels_per_pixel[i] = np.bincount(labels_in_window.astype('int')).argmax()
    
    times_and_labels = np.concatenate((times.reshape(times.shape[0],1), labels_per_pixel.reshape(labels_per_pixel.shape[0],1)), axis = 1)
    
    
    
    
    dat = {
            's': spectrogram,
            't': times, 
            'f':frequencies, 
            'labels':labels_per_pixel
            }
    
    np.savez(f'{folderpath_song}synthetic_data.npz', **dat)
    write(f'{folderpath_song}audio_representation.wav', sampling_freq, total_envelope)
    
    
    num_repeats = np.array(num_repeats_list)
    num_repeats_songs[song_index,:] = num_repeats
    
    syllables = np.array([])
    for syl_index in np.arange(syllable_phrase_order_w_repeats.shape[0]):
        syl = syllable_phrase_order_w_repeats[syl_index]
        repeats = num_repeats[syl_index]
        repeated_syllable = np.repeat(syl, repeats)
        syllables = np.concatenate((syllables, repeated_syllable))
        
    syllables = syllables.astype('int')
    
    
    df_dict = {
        'Syllable': syllables.tolist(), 
        'f_0': f_0_vector, 
        'B' : B_vector,
        'phi_0': phi_0_vector, 
        'delta_phi': delta_phi_vector, 
        'c': c_vector, 
        'Z1': Z_1_vector,
        'Z2': Z_2_vector, 
        'theta_1': theta_1_vector, 
        'theta_2': theta_2_vector,
        'T_flattened': T_vector
        }
    
    np.savez(f'{folderpath_song}acoustic_params_for_song.npz', **df_dict)


    df = pd.DataFrame(df_dict)
    
    # # plt.figure(figsize=(35, 35))
    # # sns.pairplot(df, hue = 'Syllable')
    # # # Adjust the layout to prevent clipping
    # # plt.tight_layout()
    # # plt.show()
    
    grouped_df = df.groupby('Syllable').mean()
    print(grouped_df.T)
    
    
    reducer = umap.UMAP()
    X  = df.iloc[:, 1:]
    X = X.values
    y = df.Syllable
    y = y.values
    embedding = reducer.fit_transform(X)
    
    plt.figure()
    # plt.scatter(embedding[:,0], embedding[:,1], c=y, cmap='viridis', s=50)
    
    categories = y 
    
    # Create separate scatter plots for each category
    for category in np.unique(categories):
        mask = categories == category
        plt.scatter(embedding[mask,0], embedding[mask,1], label=category, s=50)
    
    # Set plot labels and title
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    plt.title(f'UMAP Embedding of the Parameter Regimes for Each Syllable for Song_{song_index}')
    
    # Show the legend
    plt.legend()
    plt.savefig(f'{folderpath_song}UMAP_of_song.png')
    
    
    # plt.figure()
    # plt.plot(total_envelope)
    # plt.show()

# # # %% Now I want to plot the phrase durations across all phrases in our song



    phrase_duration_arr = np.array(phrase_duration_list)
    
    from scipy.stats import gaussian_kde
    
    # Create a histogram of the data
    hist, bins = np.histogram(phrase_duration_arr, bins=10, density=True)
    
    # Calculate the density curve using KDE
    density_curve = gaussian_kde(phrase_duration_arr)
    
    # Generate x values for the density curve
    x = np.linspace(phrase_duration_arr.min(), phrase_duration_arr.max(), 100)
    
    # Calculate the y values (density) for the density curve
    y = density_curve(x)
    
    plt.figure()
    
    # Plot the histogram
    plt.hist(phrase_duration_arr, bins=10, density=True, alpha=0.5, label='Histogram')
    
    # Plot the density curve
    plt.plot(x, y, color='red', label='Density Curve')
    
    # Set plot labels and title
    plt.xlabel('Phrase Duration')
    plt.ylabel('Density')
    plt.title(f'Distribution of Phrase Durations for Song')
    
    # Show the legend
    plt.legend()
    plt.savefig(f'{folderpath_song}Phrase_Durations_of_Song.png')
    
    # Save the array of phrase durations
    np.save(f'{folderpath_song}phrase_durations.npy', phrase_duration_arr)



# # # =============================================================================
# # # # GROUND TRUTH SIMULATION WAVEFORM -- Gardner supplemental
# # # =============================================================================


import wave
import numpy as np
import matplotlib.pyplot as plt

# Open the .wav file
wav_file = wave.open('/Users/AnanyaKapoor/Downloads/1108214s_sound/gardnersound4.wav', 'r')

# Get the audio file parameters
sample_width = wav_file.getsampwidth()
sample_rate = wav_file.getframerate()
num_frames = wav_file.getnframes()

# Read the audio data
audio_data = wav_file.readframes(num_frames)

# Convert the audio data to a numpy array
audio_array = np.frombuffer(audio_data, dtype=np.int16)

# Close the .wav file
wav_file.close()

# Generate the time axis
duration = num_frames / sample_rate
t_groundtruth = np.linspace(0, duration, num_frames)

# Plot the waveform
plt.figure()
plt.plot(t_groundtruth, audio_array)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Ground Truth Waveform')
plt.show()

from scipy import signal

frequencies, times, spectrogram = signal.spectrogram(audio_array, fs=sample_rate,
                                                    window='hamming', nperseg=256,
                                                    noverlap=128, nfft=512)

plt.figure()
plt.title("Tim's Synthetic Song")
plt.pcolormesh(times, frequencies, spectrogram, cmap='jet')
plt.show()


# # # =============================================================================
# # # # # Let's look at a real canary song spectrogram 
# # # =============================================================================

# import numpy as np
# dat = np.load('/Users/AnanyaKapoor/Dropbox (University of Oregon)/Kapoor_Ananya/01_Projects/01_a_Rotations/Gardner_Lab/Canary_Data/llb3/llb3_data_matrices/Python_Files/llb3_0014_2018_04_23_15_18_14.wav.npz')
# spec = dat['s']
# times = dat['t']
# frequencies = dat['f']
# labels = dat['labels']
# labels = labels.T

# plt.figure()
# plt.pcolormesh(times, frequencies, spec, cmap='jet')
# plt.show()

# import wave
# import numpy as np
# import matplotlib.pyplot as plt

# wav_file = wave.open('/Users/AnanyaKapoor/Dropbox (University of Oregon)/Kapoor_Ananya/01_Projects/01_a_Rotations/Gardner_Lab/Canary_Data/llb3/llb3_songs/llb3_0014_2018_04_23_15_18_14.wav', 'r')

# # Get the audio file parameters
# sample_width = wav_file.getsampwidth()
# sample_rate = wav_file.getframerate()
# num_frames = wav_file.getnframes()

# # Read the audio data
# audio_data = wav_file.readframes(num_frames)

# # Convert the audio data to a numpy array
# audio_array = np.frombuffer(audio_data, dtype=np.int16)

# # Close the .wav file
# wav_file.close()

# plt.figure()
# plt.plot(audio_array)
# plt.show()

# # frequencies, times, spectrogram = signal.spectrogram(audio_array, fs=sample_rate,
# #                                                     window='hamming', nperseg=256,
# #                                                     noverlap=128, nfft=1024)

# # plt.figure()
# # plt.pcolormesh(times, frequencies, spectrogram, cmap='jet')
# # plt.show()

# %% Wow let's try UMAP now


# # Parameters we set
# num_spec = 1
# window_size = 100
# stride = 10

# folderpath_song = f'{songpath}/Song_0/'
# # plt.figure()
# # plt.pcolormesh(times, frequencies, spec, cmap='jet')
# # plt.show()



# # For each spectrogram we will extract
# # 1. Each timepoint's syllable label
# # 2. The spectrogram itself
# stacked_labels = [] 
# stacked_specs = []
# # Extract the data within the numpy file. We will use this to create the spectrogram
# dat = np.load(f'{songpath}synthetic_data.npz')
# spec = dat['s']
# times = dat['t']
# frequencies = dat['f']
# labels = dat['labels']
# labels.shape = (1, labels.shape[0])
# labels = labels.T


# # Let's get rid of higher order frequencies
# mask = (frequencies<4000)&(frequencies>600)
# masked_frequencies = frequencies[mask]

# subsetted_spec = spec[mask.reshape(mask.shape[0],),:]

# stacked_labels.append(labels)
# stacked_specs.append(subsetted_spec)

    
# stacked_specs = np.concatenate((stacked_specs), axis = 1)
# stacked_labels = np.concatenate((stacked_labels), axis = 0)

# # Get a list of unique categories (syllable labels)
# unique_categories = np.unique(stacked_labels)

# # Create a dictionary that maps categories to random colors
# category_colors = {category: np.random.rand(3,) for category in unique_categories}

# spec_for_analysis = stacked_specs.T
# window_labels_arr = []
# embedding_arr = []
# # Find the exact sampling frequency (the time in miliseconds between one pixel [timepoint] and another pixel)
# dx = np.diff(times)[0]

# # We will now extract each mini-spectrogram from the full spectrogram
# stacked_windows = []
# # Find the syllable labels for each mini-spectrogram
# stacked_labels_for_window = []
# # Find the mini-spectrograms onset and ending times 
# stacked_window_times = []

# # The below for-loop will find each mini-spectrogram (window) and populate the empty lists we defined above.
# for i in range(0, spec_for_analysis.shape[0] - window_size + 1, stride):
#     # Find the window
#     window = spec_for_analysis[i:i + window_size, :]
#     # Get the window onset and ending times
#     window_times = dx*np.arange(i, i + window_size)
#     # We will flatten the window to be a 1D vector
#     window = window.reshape(1, window.shape[0]*window.shape[1])
#     # Extract the syllable labels for the window
#     labels_for_window = stacked_labels[i:i+window_size, :]
#     # Reshape the syllable labels for the window into a 1D array
#     labels_for_window = labels_for_window.reshape(1, labels_for_window.shape[0]*labels_for_window.shape[1])
#     # Populate the empty lists defined above
#     stacked_windows.append(window)
#     stacked_labels_for_window.append(labels_for_window)
#     stacked_window_times.append(window_times)

# # Convert the populated lists into a stacked numpy array
# stacked_windows = np.stack(stacked_windows, axis = 0)
# stacked_windows = np.squeeze(stacked_windows)

# stacked_labels_for_window = np.stack(stacked_labels_for_window, axis = 0)
# stacked_labels_for_window = np.squeeze(stacked_labels_for_window)

# stacked_window_times = np.stack(stacked_window_times, axis = 0)

# # For each mini-spectrogram, find the average color across all unique syllables
# mean_colors_per_minispec = np.zeros((stacked_labels_for_window.shape[0], 3))
# for i in np.arange(stacked_labels_for_window.shape[0]):
#     list_of_colors_for_row = [category_colors[x] for x in stacked_labels_for_window[i,:]]
#     all_colors_in_minispec = np.array(list_of_colors_for_row)
#     mean_color = np.mean(all_colors_in_minispec, axis = 0)
#     mean_colors_per_minispec[i,:] = mean_color


# acoustic_params_all_songs = np.empty((0, 11))

    
# # Create a list to store column vectors
# acoustic_params_columns = []

# for song_index in range(num_songs):
#     folderpath_song = f'{songpath}Song_{song_index}/'
#     acoustic_params_dat = np.load(f'{folderpath_song}acoustic_params_for_song.npz')
#     # Create a list to store column vectors
#     acoustic_params_columns = []

#     for key in acoustic_params_dat.keys():
#         array = acoustic_params_dat[key]
#         column_vector = array.reshape(-1, 1)
#         acoustic_params_columns.append(column_vector)
#     # Stack the column vectors horizontally to create the final array
#     acoustic_params_arr = np.hstack(acoustic_params_columns)    
#     acoustic_params_all_songs = np.concatenate((acoustic_params_all_songs, acoustic_params_arr))
    
# import umap

# reducer = umap.UMAP()
# X  = acoustic_params_all_songs[:,1::]
# y = acoustic_params_all_songs[:,0]
# embedding = reducer.fit_transform(X)

# plt.figure()
# # plt.scatter(embedding[:,0], embedding[:,1], c=y, cmap='viridis', s=50)

# categories = y 

# # Create separate scatter plots for each category
# for category in np.unique(categories):
#     mask = categories == category
#     plt.scatter(embedding[mask,0], embedding[mask,1], label=category, s=50)

# # Set plot labels and title
# plt.xlabel('UMAP 1')
# plt.ylabel('UMAP 2')
# plt.title(f'UMAP Embedding of the Parameter Regimes for Each Syllable across all songs')

# # Show the legend
# plt.legend()






