CUDA_VISIBLE_DEVICES=0 python main.py \
    --run_description ecgOnly \
    --model cnn \
    --pretrain_dataset ECG \
    --target_dataset PTBXL \
    --training_mode fine_tune_test \
    --wandb
