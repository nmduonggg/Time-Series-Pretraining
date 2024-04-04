CUDA_VISIBLE_DEVICES=1 python main.py \
    --run_description ecgOnly \
    --model cnn \
    --pretrain_dataset ECG \
    --target_dataset PTBXL \
    --training_mode fine_tune \
    --pretrain_lr 0.0001 \
    --lr 0.0001 \
    --wandb \
    # --use_pretrain
