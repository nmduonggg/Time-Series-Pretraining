mkdir code/TFC/data/{SleepEEG,Epilepsy,FD-A,FD-B,HAR,Gesture,ECG,EMG}
ln -s datasets/SleepEEG/{train,val,test}.pt code/TFC/data/SleepEEG/
ln -s datasets/Epilepsy/{train,val,test}.pt code/TFC/data/Epilepsy/
ln -s datasets/FD-A/{train,val,test}.pt code/TFC/data/FD-A/
ln -s datasets/FD-B/{train,val,test}.pt code/TFC/data/FD-B/
ln -s datasets/HAR/{train,val,test}.pt code/TFC/data/HAR/
ln -s datasets/Gesture/{train,val,test}.pt code/TFC/data/Gesture/
ln -s datasets/ECG/{train,val,test}.pt code/TFC/data/ECG/
ln -s datasets/EMG/{train,val,test}.pt code/TFC/data/EMG/