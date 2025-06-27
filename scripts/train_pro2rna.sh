#!/bin/bash

# FlexiblePro2RNA 训练脚本
# 基于 examples/train_flexible_pro2rna.py 进行训练和测试
# 支持多种mRNA解码器类型：mRNAGPT、GenerRNA、MLP

set -e

# ==================== 配置 ====================

# 项目根目录
PROJECT_ROOT="/home/yzhang/research/pro2rna"
cd "$PROJECT_ROOT"

# 数据路径
TRAIN_DATA="/home/yzhang/research/pro2rna/data/output/train.jsonl"
VAL_DATA="/home/yzhang/research/pro2rna/data/output/valid.jsonl"
TEST_DATA="/home/yzhang/research/pro2rna/data/output/test.jsonl"

# Decoder配置
DECODER_CONFIG="configs/decoder_configs.yaml"
DECODER_TYPE="mrna_gpt"  # 可选：mrna_gpt, generrna, mlp

# 模型配置
TAXONOMY_MODEL="allenai/scibert_scivocab_uncased"
PROTEIN_MODEL="facebook/esm2_t33_650M_UR50D"
FUSION_DIM=512

# 训练配置（可被配置文件覆盖）
BATCH_SIZE=4  # 如果设为null，将使用配置文件中的batch_size
LEARNING_RATE=null  # 如果设为null，将使用配置文件中的learning_rate
EPOCHS=5
MAX_PROTEIN_LENGTH=1024
MAX_MRNA_LENGTH=null  # 如果设为null，将使用decoder配置中的max_length

# 输出目录
OUTPUT_DIR="outputs/flexible_pro2rna_${DECODER_TYPE}_$(date +%Y%m%d_%H%M%S)"

# GPU设置
export CUDA_VISIBLE_DEVICES=5

echo "========================================"
echo "    FlexiblePro2RNA 训练测试脚本"
echo "========================================"
echo "解码器类型: $DECODER_TYPE"
echo "解码器配置: $DECODER_CONFIG"
echo "训练数据: $TRAIN_DATA"
echo "验证数据: $VAL_DATA" 
echo "测试数据: $TEST_DATA"
echo "批次大小: $BATCH_SIZE (null表示使用配置文件)"
echo "学习率: $LEARNING_RATE (null表示使用配置文件)"
echo "训练轮数: $EPOCHS"
echo "融合维度: $FUSION_DIM"
echo "输出目录: $OUTPUT_DIR"
echo "========================================"

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 构建训练命令
TRAIN_CMD="python examples/train_flexible_pro2rna.py \
    --decoder_config \"$DECODER_CONFIG\" \
    --decoder_type \"$DECODER_TYPE\" \
    --train_data \"$TRAIN_DATA\" \
    --val_data \"$VAL_DATA\" \
    --test_data \"$TEST_DATA\" \
    --taxonomy_model \"$TAXONOMY_MODEL\" \
    --protein_model \"$PROTEIN_MODEL\" \
    --fusion_dim \"$FUSION_DIM\" \
    --epochs \"$EPOCHS\" \
    --max_protein_length \"$MAX_PROTEIN_LENGTH\" \
    --output_dir \"$OUTPUT_DIR\" \
    --wandb_project \"flexible-pro2rna\" \
    --logging_steps 50 \
    --eval_steps 200 \
    --save_steps 500 \
    --num_workers 4"

# 添加可选参数（仅当不为null时）
if [ "$BATCH_SIZE" != "null" ]; then
    TRAIN_CMD="$TRAIN_CMD --batch_size $BATCH_SIZE"
fi

if [ "$LEARNING_RATE" != "null" ]; then
    TRAIN_CMD="$TRAIN_CMD --lr $LEARNING_RATE"
fi

if [ "$MAX_MRNA_LENGTH" != "null" ]; then
    TRAIN_CMD="$TRAIN_CMD --max_mrna_length $MAX_MRNA_LENGTH"
fi

echo ""
echo "执行训练命令:"
echo "$TRAIN_CMD"
echo ""

# 开始训练（包含测试）
eval "$TRAIN_CMD"

echo ""
echo "========================================"
echo "           训练测试完成!"
echo "========================================"
echo "解码器类型: $DECODER_TYPE"
echo "输出目录: $OUTPUT_DIR"
echo "最优模型: $OUTPUT_DIR/best_model.pt"
echo "最终模型: $OUTPUT_DIR/final_model.pt"
echo "词汇表: $OUTPUT_DIR/tokenizer.json"
echo "测试结果: $OUTPUT_DIR/test_results.json"
echo ""
echo "可以使用以下命令查看测试结果:"
echo "cat $OUTPUT_DIR/test_results.json | jq ."