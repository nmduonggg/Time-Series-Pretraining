CUDA_VISIBLE_DEVICES=0 python main.py \
    --run_description ecgOnly \
    --model cnn \
    --pretrain_dataset ECG \
    --target_dataset PTBXL \
    --training_mode fine_tune \
    --pretrain_lr 0.00005 \
    --lr 0.00005 \
    --use_pretrain \
    --wandb
