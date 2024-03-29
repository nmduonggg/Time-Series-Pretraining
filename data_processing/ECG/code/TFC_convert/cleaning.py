import os
import numpy as np
from tqdm import tqdm

def get_XY_from_file(fileX, fileY):
    x = np.load(fileX)
    y = np.load(fileY)
    return x, y

def get_XY_from_folder(folder):
    """
    folder:
    |-x1.npy, y1.npy
    |-...
    """
    # X, Y = list(), list()
    # num_files = len([f for f in os.listdir(folder) if ".npy" in f]) // 2
    # for i in range(num_files):
    #     pathX = os.path.join(folder, f'x{i}.npy')
    #     pathY = os.path.join(folder, f'y{i}.npy')
    #     x, y = get_XY_from_file(pathX, pathY)
    #     X.append(x)
    #     Y.append(y)

    pathX = os.path.join(folder, f'X.npy')
    pathY = os.path.join(folder, f'y.npy')
    x, y = get_XY_from_file(pathX, pathY)

    return x, y

def separate_by_labels(use_for_test):
    test_folder = os.path.join(processed_folder, use_for_test)
    X1, Y, Xn = list(), list(), list()  # single-label X/Y, multi-label X
    x, y = get_XY_from_folder(test_folder)
    
    y_sum = np.sum(y.astype(int), axis=1)
    eq_ind = np.where(y_sum==1)[0]
    ne_ind = np.where(y_sum != 1)[0]
    X1 = x[eq_ind, ...]
    Y = y[eq_ind, ...]
    Xn = x[ne_ind, ...]
            
    assert len(X1)==len(Y)
    X1 = np.stack(X1, axis=0)
    Y = np.stack(Y, axis=0)
    Xn = np.stack(Xn, axis=0)
    return X1, Y, Xn

def main():
    # single_labels_X, single_labels_Y, multi_labels_X = separate_by_labels(use_for_test)
    # print('Done processing test dataset')
    pretrain_names = [f for f in os.listdir(processed_folder) if use_for_test not in f]
    cnt = 0
    for name in pretrain_names:
        print('Processing:', name)
        X, _ = get_XY_from_folder(os.path.join(processed_folder, name))
        for i in tqdm(range(X.shape[0]), total=X.shape[0]):
            np.save(os.path.join(out_dir_pretrain, f'X{cnt}.npy'), X[i])
            cnt += 1
            
    # print('Process multi-label X: ', use_for_test)
    # multi_labels_X = multi_labels_X.reshape(-1, multi_labels_X.shape[-1])   # flatten out 12 leads
    # for i in tqdm(range(multi_labels_X.shape[0]), total=multi_labels_X.shape[0]):
    #     np.save(os.path.join(out_dir_pretrain, f'X{cnt}.npy'), multi_labels_X[i])
    #     cnt += 1        
    
    
    print('--------------------------------')
    print('Summary: %d pretrain samples' % cnt)
    print('--------------------------------')
    
    # np.save(os.path.join(out_dir_test, 'X.npy'), single_labels_X)
    # np.save(os.path.join(out_dir_test, 'Y.npy'), single_labels_Y)
    
if __name__=="__main__":
    processed_folder = "/mnt/disk1/nmduong/ECG-Pretrain/data_processing/ECG/PROCESSED"
    save_folder = "/mnt/disk1/nmduong/ECG-Pretrain/data_processing/ECG/SPECIFIC_DATA"
    use_for_test = "ptb-xl"
    out_dir = os.path.join(save_folder, "TFC")
    out_dir_pretrain = os.path.join(out_dir, "pretrain")
    out_dir_test = os.path.join(out_dir, "finetune")
    
    for od in [out_dir_pretrain, out_dir_test]:
        os.makedirs(od, exist_ok=True)
    
    main()
    
    
    