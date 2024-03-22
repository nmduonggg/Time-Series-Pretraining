CUDA_VISIBLE_DEVICES=3 python main.py \
    --run_description ecgOnly \
    --model transformer \
    --pretrain_dataset ECG \
    --target_dataset PTBXL \
    --training_mode pre_train