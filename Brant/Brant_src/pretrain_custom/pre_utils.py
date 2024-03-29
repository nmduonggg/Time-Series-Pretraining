import os
import random
from scipy import signal

import numpy as np
import torch
import torch.distributed as dist
from tqdm import tqdm


def compute_power(data, fs):
    f, Pxx_den = signal.periodogram(data, fs)

    f_thres = [4, 8, 13, 30, 50, 70, 90, 110, 128]
    poses = []
    for fi in range(len(f_thres) - 1):
        cond1_pos = np.where(f_thres[fi] < f)[0]
        cond2_pos = np.where(f_thres[fi + 1] >= f)[0]
        poses.append(np.intersect1d(cond1_pos, cond2_pos))

    ori_shape = Pxx_den.shape[:-1]
    Pxx_den = Pxx_den.reshape(-1, len(f))
    band_sum = [np.sum(Pxx_den[:, band_pos], axis=-1) + 1 for band_pos in poses]
    band_sum = [np.log10(_band_sum)[:, np.newaxis] for _band_sum in band_sum]
    band_sum = np.concatenate(band_sum, axis=-1)
    ori_shape += (8,)
    band_sum = band_sum.reshape(ori_shape)

    return band_sum


def master_save(state_dict, path):
    if dist.is_initialized():
        if dist.get_rank() == 0:
            torch.save(state_dict, path)
    else:
        torch.save(state_dict, path)


def master_print(str):
    if dist.is_initialized():
        if dist.get_rank() == 0:
            print(str, flush=True)
    else:
        print(str, flush=True)
        
def interp(x, up_win_size):
    l = x.size
    x_axis = np.linspace(1, l, l)
    up_x_axis = np.linspace(1, l, up_win_size)
    up_x = np.interp(up_x_axis, x_axis, x)
    
    return up_x
        
def _load_data_by_file_indices(file_indices, folder, cal_power, use_power):
    data, power = [], []
    for idx in file_indices:
        _data = np.load(os.path.join(folder, f'X{idx}.npy'))
        _data = interp(_data, 15*100).reshape(15, 100)
        _power = None
        
        if use_power:
            if cal_power:
                current_power = []
                for i in range(_data.shape[0]):
                    x = _data[i, :]
                    _power = compute_power(x, fs=256)
                    current_power.append(_power)
                _power = np.stack(current_power, axis=0)
            else:
                _power = np.load(os.path.join(folder, 'power.npy'))
                
        data.append(_data)
        power.append(_power)
    
    data = np.expand_dims(np.stack(data, axis=0), 0)
    power = np.expand_dims(np.stack(power, axis=0), 0)
    
    return data, power

def _load_data(file, cal_power, use_power):
    _data = np.load(os.path.join(file, 'data.npy'))

    _power = None
    if use_power:
        if cal_power:
            _power = compute_power(_data, fs=256)
        else:
            _power = np.load(os.path.join(file, 'power.npy'))

    return _data, _power


def load_data(files, cal_power, use_power=True):
    data, power = [], []
    for file in tqdm(files, disable=True):
        _data, _power = _load_data(file, cal_power=cal_power, use_power=use_power)

        data.append(_data)
        power.append(_power)

    if not use_power:
        power = None
    return data, power


def generate_mask(ch_num, seq_len, mask_ratio):
    mask_num = int(ch_num*seq_len*mask_ratio)
    pos = list(range(ch_num*seq_len))

    return random.sample(pos, mask_num)


def get_loss(mask_loss):
    if dist.is_initialized():
        mask_loss_list = [torch.zeros(1).to(dist.get_rank()) for _ in range(dist.get_world_size())]
        dist.all_gather(mask_loss_list, mask_loss)

        return sum(mask_loss_list).item()
    else:
        return mask_loss

