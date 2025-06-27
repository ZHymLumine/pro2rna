#!/bin/bash
export CUDA_LAUNCH_BLOCKING=1
export WANDB_API_KEY="986353ee165f067cdb44f4be649d449151ca9ad2"
export TOKENIZERS_PARALLELISM=false

export CUDA_VISIBLE_DEVICES=0

NUM_NODES=1  # number of node
NODE_RANK=0  # master node: 0
MASTER_ADDR="192.168.0.42"  # IP address of node 0
MASTER_PORT=12345
GPU_PER_NODE=1

torchrun --nnodes=$NUM_NODES --nproc_per_node=$GPU_PER_NODE --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
    ../protein2rna/train_t5.py \
    --train_data_path /raid_elmo/home/lr/zym/protein2rna/ncbi_dataset/data/train_flat.json \
    --valid_data_path /raid_elmo/home/lr/zym/protein2rna/ncbi_dataset/data/valid_flat.json \
    --output_dir /raid_elmo/home/lr/zym/protein2rna/checkpoints/codont5_2_tokenizer \
    --model_name "t5-base" \
    --num_train_epochs 20 \
    --batch_size 8 \
    --eval_strategy "epoch" \
    --fp16 \
    --gradient_accumulation_steps 8 \
    --save_steps 5000 \
    --save_total_limit 1 \
    --learning_rate 1e-4 \
    --weight_decay 0.01 \
    --warmup_steps 1000 \
    --logging_steps 100 \
    --report_to wandb \
    --run_name "CodonT5"