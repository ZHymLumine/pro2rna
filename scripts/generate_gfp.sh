#!/bin/bash

export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=4

cd /home/yzhang/research/pro2rna

python scripts/generate_gfp.py \
    --model_path /home/yzhang/research/pro2rna/checkpoints/esm2_650M_scibert_mRNAGPT/checkpoint-539/pytorch_model.bin \
    --esm_name_or_path "esm2_t33_650M_UR50D" \
    --species_model "scibert" \
    --decoder_type "RNAdecoder" \
    --RNA_config_path /home/yzhang/research/pro2rna/pro2rna/mRNAGPT/mrnagpt_config.json \
    --decoder_path /home/yzhang/research/pro2rna/pro2rna/mRNAGPT/ckpt_563000.pt \
    --protein_fasta /home/yzhang/research/pro2rna/data/downstream/gfp_protein_use.faa \
    --species_csv /home/yzhang/research/pro2rna/data/downstream/human_filtered.csv \
    --output_dir /home/yzhang/research/pro2rna/data/downstream/gfp_output \
    --num_samples 50 \
    --temperature 0.8 \
    --top_k 50 \
    --top_p 0.9 \
    --max_length 512


