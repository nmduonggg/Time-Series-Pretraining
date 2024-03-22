import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import os
import numpy as np
from augmentations import *
import torch.fft as fft

class ECGDataset(Dataset):
    # Initialize your data, download, etc.
    # Pretrain: y_train = None, y_train only get value in finetune_mode
    def __init__(self, folder, config, training_mode, train, target_dataset_size=64, subset=False):
        super(ECGDataset, self).__init__()
        self.training_mode = training_mode
        self.train = train
        
        # The folder can contain both forms:
        # |-X1.npy, X2.npy, etc. (each for 1 sample)
        # |-X.npy (all samples)
        
        X_train = np.load(os.path.join(folder, 'X_train.npy' if self.train else 'X_test.npy'))
        if training_mode != 'pre_train':
            y_train = np.load(os.path.join(folder, 'y_train.npy' if self.train else 'y_test.npy'))
        else:
            y_train = None 
    
        
        # """Align the TS length between source and target datasets"""
        # X_train = X_train[:, :, :int(config.TSlength_aligned)] # take the first 178 samples

        """Subset for debugging"""
        if subset == True:
            subset_size = target_dataset_size * 10 #30 #7 # 60*1
            """if the dimension is larger than 178, take the first 178 dimensions. If multiple channels, take the first channel"""
            X_train = X_train[:subset_size, ...]
            y_train = y_train[:subset_size, ...] if y_train is not None else None
            print('Using subset for debugging, the datasize is:', y_train.shape[0])

        if isinstance(X_train, np.ndarray):
            self.x_data = torch.from_numpy(X_train)
            self.y_data = torch.from_numpy(y_train).long() if y_train is not None else None
        else:
            self.x_data = X_train
            self.y_data = y_train
            
        self.x_data = self.x_data.unsqueeze(1)  # to [Nx1x1000]
        # print(self.x_data.shape)

        """Transfer x_data to Frequency Domain. If use fft.fft, the output has the same shape; if use fft.rfft, 
        the output shape is half of the time window."""

        window_length = self.x_data.shape[-1]
        self.x_data_f = fft.fft(self.x_data).abs() #/(window_length) # rfft for real value inputs.
        self.len = X_train.shape[0]

        """Augmentation"""
        if training_mode == "pre_train":  # no need to apply Augmentations in other modes
            self.aug1 = DataTransform_TD_bank(self.x_data, config)
            self.aug1_f = DataTransform_FD(self.x_data_f, config) # [7360, 1, 90]

    def __getitem__(self, index):
        if self.training_mode == "pre_train":
            return self.x_data[index], self.y_data, self.aug1[index],  \
                   self.x_data_f[index], self.aug1_f[index]
        else:
            return self.x_data[index], self.y_data[index], self.x_data[index], \
                   self.x_data_f[index], self.x_data_f[index]

    def __len__(self):
        return self.len

class ECGPretrain(Dataset):
    # Initialize your data, download, etc.
    # Pretrain: y_train = None, y_train only get value in finetune_mode
    def __init__(self, folder, config, training_mode, train, target_dataset_size=64, subset=False):
        super(ECGPretrain, self).__init__()
        self.training_mode = training_mode
        assert training_mode == "pre_train"
        self.train = train
        self.folder = folder
        
        # The folder can contain both forms:
        # |-X1.npy, X2.npy, etc. (each for 1 sample)
        # |-X.npy (all samples)
        
        
        self.len = 6300000  # faster loading
        self.y_data = None
        self.config = config
        
        # """Align the TS length between source and target datasets"""
        # X_train = X_train[:, :, :int(config.TSlength_aligned)] # take the first 178 samples

        """Subset for debugging"""
        if subset == True:
            subset_size = target_dataset_size * 10 #30 #7 # 60*1
            """if the dimension is larger than 178, take the first 178 dimensions. If multiple channels, take the first channel"""
            self.len = subset_size
            print('Using subset for debugging, the datasize is:', self.len)

    def __getitem__(self, index):
        if self.training_mode == "pre_train":
            x_path = os.path.join(self.folder, f'X{index}.npy')
            x = np.load(x_path)
            
            if isinstance(x, np.ndarray):
                x = torch.from_numpy(x)
            else:
                x = x
            x = x.reshape(1, 1, -1)  # 1x1x1000
            # print(x.shape)
            x_f = fft.fft(x).abs()
            aug1 = DataTransform_TD_bank(x, self.config)
            aug1_f = DataTransform_FD(x_f, self.config)
            self.y_data = torch.zeros(1)
            return x.squeeze(0), self.y_data, aug1.squeeze(0), x_f.squeeze(0), aug1_f.squeeze(0)

    def __len__(self):
        return self.len

def data_generator(sourcedata_path, targetdata_path, configs, training_mode, subset=True):

    # subset = True # if true, use a subset for debugging.
    train_loader, finetune_loader, test_loader = None, None, None
    if training_mode=='pre_train':
        train_dataset = ECGPretrain(sourcedata_path, configs, training_mode, True, target_dataset_size=configs.batch_size, subset=subset) # for self-supervised, the data are augmented here
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=configs.batch_size,
                                               shuffle=True, drop_last=configs.drop_last,
                                               num_workers=0)
    
    if training_mode!='pre_train':
        finetune_dataset = ECGDataset(targetdata_path, configs, training_mode, True, target_dataset_size=configs.target_batch_size, subset=subset)
        test_dataset = ECGDataset(targetdata_path, configs, training_mode, False,
                                    target_dataset_size=configs.target_batch_size, subset=subset)

        
        finetune_loader = torch.utils.data.DataLoader(dataset=finetune_dataset, batch_size=configs.target_batch_size,
                                                shuffle=True, drop_last=configs.drop_last,
                                                num_workers=0)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=configs.target_batch_size,
                                                shuffle=True, drop_last=configs.drop_last,
                                                num_workers=0)
    
    return train_loader, finetune_loader, test_loader
