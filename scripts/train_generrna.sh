#!/bin/bash

# FlexiblePro2RNA - GenerRNA解码器训练脚本
# 专门用于训练GenerRNA decoder (词汇表大小1024，k-mer级别)

set -e

# ==================== 配置 ====================

# 项目根目录
PROJECT_ROOT="/home/yzhang/research/pro2rna"
cd "$PROJECT_ROOT"

# 数据路径
TRAIN_DATA="/home/yzhang/research/pro2rna/data/output/train.jsonl"
VAL_DATA="/home/yzhang/research/pro2rna/data/output/valid.jsonl"
TEST_DATA="/home/yzhang/research/pro2rna/data/output/test.jsonl"

# GenerRNA Decoder配置
DECODER_CONFIG="configs/decoder_configs.yaml"
DECODER_TYPE="generrna"

# 模型配置
TAXONOMY_MODEL="allenai/scibert_scivocab_uncased"
PROTEIN_MODEL="facebook/esm2_t33_650M_UR50D"
FUSION_DIM=512

# GenerRNA特定训练配置
BATCH_SIZE=8        # GenerRNA可以支持更大batch size
LEARNING_RATE=5e-5  # 稍低的学习率适合更大的词汇表
EPOCHS=5
MAX_PROTEIN_LENGTH=1024
MAX_MRNA_LENGTH=256   # GenerRNA较短的最大长度

# 输出目录
OUTPUT_DIR="outputs/generrna_$(date +%Y%m%d_%H%M%S)"

# GPU设置
export CUDA_VISIBLE_DEVICES=5

echo "========================================"
echo "      GenerRNA Decoder 训练脚本"
echo "========================================"
echo "解码器类型: $DECODER_TYPE (k-mer级别, vocab_size=1024)"
echo "预训练模型: /raid_elmo/home/lr/zym/protein2rna/GenerRNA/model.pt.recombined"
echo "训练数据: $TRAIN_DATA"
echo "验证数据: $VAL_DATA" 
echo "测试数据: $TEST_DATA"
echo "批次大小: $BATCH_SIZE"
echo "学习率: $LEARNING_RATE"
echo "训练轮数: $EPOCHS"
echo "融合维度: $FUSION_DIM"
echo "最大mRNA长度: $MAX_MRNA_LENGTH"
echo "输出目录: $OUTPUT_DIR"
echo "========================================"

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 开始训练
python examples/train_flexible_pro2rna.py \
    --decoder_config "$DECODER_CONFIG" \
    --decoder_type "$DECODER_TYPE" \
    --train_data "$TRAIN_DATA" \
    --val_data "$VAL_DATA" \
    --test_data "$TEST_DATA" \
    --taxonomy_model "$TAXONOMY_MODEL" \
    --protein_model "$PROTEIN_MODEL" \
    --fusion_dim "$FUSION_DIM" \
    --batch_size "$BATCH_SIZE" \
    --lr "$LEARNING_RATE" \
    --epochs "$EPOCHS" \
    --max_protein_length "$MAX_PROTEIN_LENGTH" \
    --max_mrna_length "$MAX_MRNA_LENGTH" \
    --output_dir "$OUTPUT_DIR" \
    --wandb_project "flexible-pro2rna-generrna" \
    --logging_steps 50 \
    --eval_steps 200 \
    --save_steps 500 \
    --num_workers 4

echo ""
echo "========================================"
echo "        GenerRNA训练完成!"
echo "========================================"
echo "解码器类型: $DECODER_TYPE"
echo "词汇表大小: 1024 (k-mer级别)"
echo "输出目录: $OUTPUT_DIR"
echo "最优模型: $OUTPUT_DIR/best_model.pt"
echo "测试结果: $OUTPUT_DIR/test_results.json"
echo ""
echo "查看测试结果:"
echo "cat $OUTPUT_DIR/test_results.json | jq .sample_sequences" 