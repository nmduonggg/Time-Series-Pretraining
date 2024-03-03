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

    data = [] # 14 features: one feature for each lead, one feature for age, and one feature for sex
    onehot_labels = np.zeros((num_recordings, num_classes), dtype=np.bool_) # One-hot encoding of classes

    for i in range(num_recordings):

        # Load header and recording.
        header = load_header(header_files[i])
        recording = load_recording(recording_files[i])  
        _, n_samples = recording.shape
        
        # real num samples > min num samples -> divide
        
        if n_samples > num_samples:
            num_segments = int(n_samples // num_samples)
            for i in range(num_segments-1):
                data.append(recording[:, int(i*num_samples): int(i*num_samples+num_samples)])
                
        else:
            data.append(recording)

        current_labels = get_labels(header)
        for label in current_labels:
            if label in classes:
                j = classes.index(label)
                onehot_labels[i, j] = 1
    data = np.stack(data, axis=0)
    
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
    
    return datas   

def main():
    raw_dir_path = '/mnt/disk4/nmduong/Time-Series-Pretraining/data_processing/ECG/RAW_DATA/files/challenge-2021/1.0.3/training'
    save_dir_path = '/mnt/disk4/nmduong/Time-Series-Pretraining/data_processing/ECG/PROCESSED'
    os.makedirs(save_dir_path, exist_ok=True)
    all_data_dirs = [os.path.join(raw_dir_path, r) for r in os.listdir(raw_dir_path) if os.path.isdir(os.path.join(raw_dir_path, r))]
    
    cnt = 0
    for dt_dir in all_data_dirs:
        dt_name = dt_dir.split('/')[-1]
        print('Processing %s ...'%dt_dir)
        current_data_list = process_parent_dir(dt_dir)
        
        for i, current_data in enumerate(current_data_list):
            save_path = os.path.join(save_dir_path, f'{dt_name}')
            os.makedirs(save_path, exist_ok=True)
            save_path = os.path.join(save_path, f'x{i}.npy')
            np.save(save_path, current_data)
            
            cnt += current_data.shape[0]
    
    print("Detect and process %d samples with multiple leads" % cnt)
        
        
if __name__ == '__main__':
    main()