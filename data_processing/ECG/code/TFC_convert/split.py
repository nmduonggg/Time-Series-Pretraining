import os
import numpy as np

# index

def onehot2label(onehot):
    return np.argmax(onehot, axis=1)

def separate_by_labels(x, y):
    X1, Y, Xn = list(), list(), list()  # single-label X/Y, multi-label X
    
    y_sum = np.sum(y.astype(int), axis=1)
    eq_ind = np.where(y_sum==1)[0]
    ne_ind = np.where(y_sum != 1)[0]
    X1 = x[eq_ind, ...]
    Y = y[eq_ind, ...]
    Xn = x[ne_ind, ...]
            
    assert len(X1)==len(Y)
    # X1 = np.stack(X1, axis=0)
    # Y = np.stack(Y, axis=0)
    # Xn = np.stack(Xn, axis=0)
    return X1, Y, Xn


def main():
    x = np.load(X_path)
    y = np.load(y_path)
    
    # x = x.reshape(-1, x.shape[-1])
    # y = y.reshape(-1, y.shape[-1])
    N, C, T = x.shape
    
    x, y, xn = separate_by_labels(x, y)
    print(x.shape, y.shape)
    index_y = np.argmax(y, axis=-1)
    y = np.repeat(np.expand_dims(y, axis=1), 12, axis=1)
    datas = np.concatenate((x, y), axis=-1)
    max_num_classes = y.shape[-1]

    new_classes = sorted([i for i in set(index_y)])
    
    new_new_classes = []
    for c in range(max_num_classes):
        if c not in new_classes: continue
        ind_c = np.where(index_y==c)[0]
        datas_c = datas[ind_c, ...]
        if datas_c.shape[0] >= 2:
            new_new_classes.append(c)
        else: continue
        
    new_classes = new_new_classes
    
    print(new_classes)
    
    print('Original num classes %d'%max_num_classes)
    
    # y = y[:, :max_num_classes+1]
    
    
    X_train, Y_train, X_test, Y_test = list(), list(), list(), list()
    train_cnt = 0
    test_cnt = 0
    
    print('Detect %d classes in total' % len(new_classes))
    for c in range(max_num_classes):
        if c not in new_classes: continue
        
        ind_c = np.where(index_y==c)[0]
        datas_c = datas[ind_c, ...]
        test_num = max(1, int(test_ratio*datas_c.shape[0]))
        train_num = datas_c.shape[0]-test_num
        print('Split %d for train out of %d samples in total for class %d' % (train_num, datas_c.shape[0], c))
        
        np.random.shuffle(datas_c)
        X_train.append(datas_c[:train_num, :, :T])
        Y_train.append(np.ones(train_num)*new_classes.index(c))
        X_test.append(datas_c[train_num:, :, :T])
        Y_test.append(np.ones(test_num)*new_classes.index(c))
        
        train_cnt += train_num
        test_cnt += test_num
    
    print('Detect %d train samples, %d test samples' % (train_cnt, test_cnt))
    
    print(np.concatenate(Y_train).max())
    print(np.concatenate(Y_test).max())
        
    np.save(X_train_path, np.concatenate(X_train, axis=0))
    np.save(y_train_path, np.concatenate(Y_train, axis=0))
    np.save(X_test_path, np.concatenate(X_test, axis=0))
    np.save(y_test_path, np.concatenate(Y_test, axis=0))
    
    

if __name__ == '__main__':
    X_path = "/mnt/disk1/nmduong/ECG-Pretrain/data_processing/ECG/PROCESSED/ptb-xl/X.npy"
    y_path = "/mnt/disk1/nmduong/ECG-Pretrain/data_processing/ECG/PROCESSED/ptb-xl/y.npy"
    X_train_path = '/mnt/disk1/nmduong/ECG-Pretrain/data_processing/ECG/PROCESSED/ptb-xl/X_train.npy'
    y_train_path = '/mnt/disk1/nmduong/ECG-Pretrain/data_processing/ECG/PROCESSED/ptb-xl/y_train.npy'
    X_test_path = '/mnt/disk1/nmduong/ECG-Pretrain/data_processing/ECG/PROCESSED/ptb-xl/X_test.npy'
    y_test_path = '/mnt/disk1/nmduong/ECG-Pretrain/data_processing/ECG/PROCESSED/ptb-xl/y_test.npy'
    test_ratio = 0.2
    
    main()