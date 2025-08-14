#!/bin/bash

# Multi-GPU training script for 5 GPUs
# Make sure you have 5 GPUs available

export CUDA_VISIBLE_DEVICES=0,1,2,3,4

python training.py \
    --train_data_path "processed_dialogues.txt" \
    --output_dir "./jasper_model_checkpoints_5gpu" \
    --model_name "bert-base-uncased" \
    --num_epochs 100 \
    --batch_size 512 \
    --max_len 512 \
    --learning_rate 5e-5 \
    --warmup_steps 1000 \
    --save_every 5 \
    --teacher_update_every 10 \
    --cosine_loss_weight 10.0 \
    --similarity_loss_weight 200.0 \
    --contrastive_loss_weight 20.0 \
    --contrastive_margin 0.5 \
    --world_size 5 \
    --wandb_project "dialog-mteb-pretrain-5gpu" \
    --wandb_entity "your_wandb_entity" 