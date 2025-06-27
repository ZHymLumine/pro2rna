CUDA_VISIBLE_DEVICES=0 /home/lr/zym/.pyenv/versions/anaconda3-2023.09-0/envs/protein2rna/bin/python evaluate_model.py \
    --dataset_path /raid_elmo/home/lr/zym/protein2rna/ncbi_dataset/data/GCF/ \
    --output_dir /raid_elmo/home/lr/zym/protein2rna/results/esm2_35M_scibert_mRNAdesigner \
    --esm_name_or_path "esm2_t12_35M_UR50D" \
    --model_path /raid_elmo/home/lr/zym/protein2rna/checkpoints/esm2_35M_scibert_mRNAdesigner/checkpoint-44000/pytorch_model.bin  \
    --species_model "scibert" \
    --decoder_type "RNAdecoder" \
    --RNA_config_path /home/lr/zym/research/protein2rna/protein2rna/mRNAdesigner/config.json \
    --decoder_path /home/lr/zym/research/protein2rna/protein2rna/mRNAdesigner/ckpt_2000.pt \
