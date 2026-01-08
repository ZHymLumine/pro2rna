export WANDB_API_KEY="986353ee165f067cdb44f4be649d449151ca9ad2"

export TOKENIZERS_PARALLELISM=false

export CUDA_VISIBLE_DEVICES=4

num_gpus=1

# decoder: mrnagpt without species model
python pro2rna/training.py \
    --dataset_path /home/yzhang/research/pro2rna/data/build \
    --output_dir /home/yzhang/research/pro2rna/checkpoints/esm2_650M_no_species_mRNAGPT \
    --esm_name_or_path "esm2_t33_650M_UR50D" \
    --species_model None \
    --num_train_epochs 20 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --eval_strategy "epoch" \
    --gradient_accumulation_steps 8 \
    --save_strategy "epoch" \
    --save_steps 20 \
    --save_total_limit 1 \
    --learning_rate 6e-4 \
    --weight_decay 0. \
    --lr_scheduler_type "cosine" \
    --dataloader_num_workers 6 \
    --logging_steps 1 \
    --report_to wandb \
    --run_name "esm2_650M_no_species_mRNAGPT_with_eval" \
    --load_best_model_at_end True \
    --metric_for_best_model "eval_loss" \
    --greater_is_better False \
    --logging_first_step True \
    --decoder_type "RNAdecoder" \
    --RNA_config_path /home/yzhang/research/pro2rna/pro2rna/mRNAGPT/mrnagpt_config.json \
    --decoder_path /home/yzhang/research/pro2rna/pro2rna/mRNAGPT/pretrained_bacteria_ckpt_563000.pt \
