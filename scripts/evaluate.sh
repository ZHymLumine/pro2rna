CUDA_VISIBLE_DEVICES=0 /home/lr/zym/.pyenv/versions/anaconda3-2023.09-0/envs/protein2rna/bin/python evaluate_model.py \
    --dataset_path /raid_elmo/home/lr/zym/protein2rna/ncbi_dataset_copy/data/GCF/ \
    --output_dir /raid_elmo/home/lr/zym/protein2rna/results/esm2_35M_mixed \
    --esm_name_or_path "esm2_t12_35M_UR50D" \
    --model_path /raid_elmo/home/lr/zym/protein2rna/checkpoints/esm2_35M_mixed_calm/checkpoint-174000/pytorch_model.bin \
    --species_model "scibert" \
    --decoder_type "mlp" \
    --optimization \

# /raid_elmo/home/lr/zym/protein2rna/checkpoints/esm2_35M_20epoch/checkpoint-44000/pytorch_model.bin

#/raid_elmo/home/lr/zym/protein2rna/checkpoints/esm2_35M_5epoch/checkpoint-88000/pytorch_model.bin

# /raid_elmo/home/lr/zym/protein2rna/checkpoints/esm2_35M_seed_1234/checkpoint-88000