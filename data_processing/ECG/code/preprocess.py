from helper_code import *
import numpy as np, os, sys, joblib
from tqdm import tqdm 

twelve_leads = ('I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6')

def process_single_dir(data_directory):

    header_files, recording_files = find_challenge_files(data_directory)
    num_recordings = len(recording_files)

    if not num_recordings:
        raise Exception('No data was provided.')

    # Extract the classes from the dataset.
    classes = set()
    num_samples = 1e9
    for header_file in header_files:
        header = load_header(header_file)
        classes |= set(get_labels(header))
        num_samples = min(num_samples, get_num_samples(header))
    if all(is_integer(x) for x in classes):
        classes = sorted(classes, key=lambda x: int(x)) # Sort classes numerically if numbers.
    else:
        classes = sorted(classes) # Sort classes alphanumerically if not numbers.
    num_classes = len(classes)

    # Extract the features and labels from the dataset.

    data = np.zeros((num_recordings, 12, int(num_samples)), dtype=np.float32) # 14 features: one feature for each lead, one feature for age, and one feature for sex
    onehot_labels = np.zeros((num_recordings, num_classes), dtype=np.bool_) # One-hot encoding of classes

    for i in range(num_recordings):

        # Load header and recording.
        header = load_header(header_files[i])
        recording = load_recording(recording_files[i])

        data[i] = recording

        current_labels = get_labels(header)
        for label in current_labels:
            if label in classes:
                j = classes.index(label)
                onehot_labels[i, j] = 1
                
    return data, onehot_labels

def process_parent_dir(parent_directory):
    """
    parent_directory 
    |- data_directory 1
        |- .hea
        |- .mat
    |- data directory 1
    """
    datas = []
    labels = []
    data_directories = [os.path.join(parent_directory, d) for d in os.listdir(parent_directory) if os.path.isdir(os.path.join(parent_directory, d))]
    for data_dir in tqdm(data_directories, total=len(data_directories)):
        X, y = process_single_dir(data_dir)
        datas.append(X)
        labels.append(y)
    datas = np.concatenate(datas)
    labels = np.concatenate(labels)
    
    print(datas.shape)
    print(labels.shape)
        
    
if __name__ == '__main__':
    data_directory = "/mnt/disk4/nmduong/Time-Series-Pretraining/data_processing/ECG/RAW_DATA/files/challenge-2021/1.0.3/training/chapman_shaoxing"
    process_parent_dir(data_directory)