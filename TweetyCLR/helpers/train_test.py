import os
import numpy as np
import shutil
from sklearn.model_selection import train_test_split

# Path where the .npz files are stored
path_of_npz_files = '/Users/AnanyaKapoor/Desktop/llb16_data_matrices'

# Create directories for training and testing if they do not exist
train_dir = os.path.join(path_of_npz_files, 'canary_train')
test_dir = os.path.join(path_of_npz_files, 'canary_test')

os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# List all .npz files in the directory
npz_files = [f for f in os.listdir(path_of_npz_files) if f.endswith('.npz')]

# Split the files into training and testing sets (80% train, 20% test)
train_files, test_files = train_test_split(npz_files, test_size=0.2, random_state=295)

# Function to move files to the specified directory
def move_files(file_list, target_dir):
    for file in file_list:
        src_path = os.path.join(path_of_npz_files, file)
        dest_path = os.path.join(target_dir, file)
        shutil.move(src_path, dest_path)
        print(f'Moved {file} to {target_dir}')

# Move the files to their respective directories
move_files(train_files, train_dir)
move_files(test_files, test_dir)

print("Files have been divided into training and testing folders.")


