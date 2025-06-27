CUDA_VISIBLE_DEVICES=0 /home/lr/zym/.pyenv/versions/anaconda3-2023.09-0/envs/protein2rna/bin/python evaluate_model.py \
    --dataset_path /raid_elmo/home/lr/zym/protein2rna/ncbi_dataset_copy/data/GCF/ \
    --output_dir /raid_elmo/home/lr/zym/protein2rna/results/esm2_35M_20epoch_mixed_no_calm \
    --esm_name_or_path "esm2_t12_35M_UR50D" \
    --model_path /raid_elmo/home/lr/zym/protein2rna/checkpoints/esm2_35M_mixed/checkpoint-25000/pytorch_model.bin \
    --species_model "scibert" \
    --decoder_type "mlp" \

# /raid_elmo/home/lr/zym/protein2rna/checkpoints/esm2_35M_20epoch/checkpoint-44000/pytorch_model.bin
# /raid_elmo/home/lr/zym/protein2rna/checkpoints/esm2_35M_20epoch_no_calm/checkpoint-88000
# /raid_elmo/home/lr/zym/protein2rna/checkpoints/esm2_35M_10epoch_no_calm

#/raid_elmo/home/lr/zym/protein2rna/checkpoints/esm2_35M_5epoch/checkpoint-14000/pytorch_model.bin
# 
#  --model_path /raid_elmo/home/lr/zym/protein2rna/checkpoints/esm2_35M_20epoch_no_calm_2/checkpoint-90000/pytorch_model.bin \