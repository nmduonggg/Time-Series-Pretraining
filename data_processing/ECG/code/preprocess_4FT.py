"""
Preprocess finetune folder in Will2Do datasets, including:
    - Cut into segments 1000 in lengths and store in corresponding folders
    - Only process exlcuded folder in preprocess.py
"""

from helper_code import *
import numpy as np, os, sys, joblib
from tqdm import tqdm 

twelve_leads = ('I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6')

def interp(x, up_win_size):
    # y = seg_meta['category_id']
    l = x.size
    x_axis = np.linspace(1, l, l)
    up_x_axis = np.linspace(1, l, up_win_size)
    up_x = np.interp(up_x_axis, x_axis, x)
    return up_x

def get_full_classes(parent_dir):
    
    classes = set()
    for data_directory in os.listdir(parent_dir):
        data_directory = os.path.join(parent_dir, data_directory)
        if not os.path.isdir(data_directory): continue
        
        header_files, recording_files = find_challenge_files(data_directory)
        num_recordings = len(recording_files)

        if not num_recordings:
            raise Exception('No data was provided.')

        for header_file in header_files:
            header = load_header(header_file)
            classes |= set(get_labels(header))
            # num_samples = min(num_samples, get_num_samples(header))
            
    if all(is_integer(x) for x in classes):
        classes = sorted(classes, key=lambda x: int(x)) # Sort classes numerically if numbers.
    else:
        classes = sorted(classes) # Sort classes alphanumerically if not numbers.
             
    return classes

def process_single_dir(data_directory, full_classes):
    header_files, recording_files = find_challenge_files(data_directory)
    num_recordings = len(recording_files)

    if not num_recordings:
        raise Exception('No data was provided.')

    # Extract the classes from the dataset.
    num_samples = 1000
    for header_file in header_files:
        header = load_header(header_file)
        # num_samples = min(num_samples, get_num_samples(header))
        
    full_classes = list(full_classes)
    num_classes = len(full_classes)

    # Extract the features and labels from the dataset.

    data = [] # 14 features: one feature for each lead, one feature for age, and one feature for sex
    labels = []
    

    for i in range(num_recordings):
        # initialize
        onehot_labels = np.zeros((num_classes), dtype=np.bool_) # One-hot encoding of classes
        current_data_list = []
        # Load header and recording.
        header = load_header(header_files[i])
        recording = load_recording(recording_files[i])  
        current_labels = get_labels(header)
        for label in current_labels:
            if label in full_classes:
                j = full_classes.index(label)
                onehot_labels[j] = 1
        
        _, n_samples = recording.shape
        
        # real num samples > min num samples -> divide
        
        assert n_samples>=num_samples, '%d - %d'%(n_samples, num_samples)
        if n_samples >= num_samples:
            with_channel = recording[:, :num_samples]
            current_data_list.append(with_channel)  # keeps all 12 leads, split later
        
        data += current_data_list
        labels += [onehot_labels for _ in current_data_list]
                
    data = np.stack(data, axis=0)
    onehots= np.stack(labels, axis=0)
    
    return data, onehots

def process_parent_dir(parent_directory):
    """
    parent_directory 
    |- data_directory 1
        |- .hea
        |- .mat
    |- data directory 2
    """
    full_classes = get_full_classes(parent_directory)
    datas = []
    labels = []
    data_directories = [os.path.join(parent_directory, d) for d in os.listdir(parent_directory) if os.path.isdir(os.path.join(parent_directory, d))]
    for data_dir in tqdm(data_directories, total=len(data_directories)):
        X, y = process_single_dir(data_dir, full_classes)
        datas.append(X)
        labels.append(y)
    assert len(datas)==len(labels)
    datas = np.concatenate(datas, axis=0)
    labels = np.concatenate(labels, axis=0)
    
    return datas, labels   

def main():
    os.makedirs(save_dir_path, exist_ok=True)
    # all_data_dirs = [os.path.join(raw_dir_path, r) for r in os.listdir(raw_dir_path) if os.path.isdir(os.path.join(raw_dir_path, r))]
    all_data_dirs = [os.path.join(raw_dir_path, exclude_name)]
    
    cnt = 0
    for dt_dir in all_data_dirs:
        # if exclude_name in dt_dir:
        #     continue
        dt_name = dt_dir.split('/')[-1]
        print('Processing %s ...'%dt_dir)
        datas, labels = process_parent_dir(dt_dir)
        # current_data_list = zip(datas, labels)
        
        # for i, current_pair in enumerate(current_data_list):
        #     current_data, current_label = current_pair
        #     save_path = os.path.join(save_dir_path, f'{dt_name}')
        #     os.makedirs(save_path, exist_ok=True)
        #     data_save_path = os.path.join(save_path, f'x{i}.npy')
        #     label_save_path = os.path.join(save_path, f'y{i}.npy')
        #     np.save(data_save_path, current_data)
        #     np.save(label_save_path, current_label)
            
        #     cnt += current_data.shape[0]
        
        save_path = os.path.join(save_dir_path, f'{dt_name}')
        os.makedirs(save_path, exist_ok=True)
        data_save_path = os.path.join(save_path, f'X.npy')
        label_save_path = os.path.join(save_path, f'y.npy')
        print("X size: ", datas.shape)
        print("Y size: ", labels.shape)
        np.save(data_save_path, datas)
        np.save(label_save_path, labels)
        
        cnt += datas.shape[0]
    
    print("Detect and process %d samples with multiple leads" % cnt)
        
        
if __name__ == '__main__':
    raw_dir_path = '/mnt/disk1/nmduong/ECG-Pretrain/data_processing/ECG/RAW_DATA/Will2Do'
    save_dir_path = '/mnt/disk1/nmduong/ECG-Pretrain/data_processing/ECG/PROCESSED'
    exclude_name = 'ptb-xl'
    print('-'*60)
    print("Raw dataset from dir %s" % raw_dir_path)
    print("Saving processed data to dir %s" % save_dir_path)
    print("Exclude interpolation in dataset %s" % exclude_name)
    print('-'*60)
    main()
    
"""
X size:  (109185, 12, 1000)
Y size:  (109185, 12, 50)
Detect and process 109185 samples with multiple leads
"""