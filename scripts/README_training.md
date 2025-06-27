# FlexiblePro2RNA 训练脚本使用指南

## 概述

FlexiblePro2RNA框架支持多种mRNA解码器类型，每种都有其特定的应用场景和优势。我们提供了针对不同decoder类型的专用训练脚本。

## 可用的解码器类型

### 1. mRNAGPT Decoder
- **文件**: `train_mrna_gpt.sh`
- **特点**: 基于mRNAdesigner预训练模型
- **词汇表**: 69 (codon级别)
- **优势**: 
  - 利用预训练知识，生成质量高
  - LoRA微调，内存需求较小
  - 适合生成符合生物学规律的mRNA序列
- **适用场景**: 生产环境，需要高质量mRNA序列

### 2. GenerRNA Decoder  
- **文件**: `train_generrna.sh`
- **特点**: 基于GenerRNA预训练模型
- **词汇表**: 1024 (k-mer级别)
- **优势**:
  - 支持更复杂的序列模式
  - 核苷酸级别的精确控制
  - 较大词汇表提供更多表达能力
- **适用场景**: 需要精确控制mRNA结构的应用

### 3. MLP Decoder
- **文件**: `train_mlp.sh`  
- **特点**: 轻量级全连接网络
- **词汇表**: 69 (codon级别)
- **优势**:
  - 训练速度快，适合快速实验
  - 模型小，部署方便
  - 易于理解和调试
- **适用场景**: 快速原型开发，基线对比

## 使用方法

### 方法1: 交互式启动 (推荐)
```bash
# 启动训练脚本管理器
bash scripts/run_training.sh
```

### 方法2: 直接运行特定脚本
```bash
# 训练mRNAGPT
bash scripts/train_mrna_gpt.sh

# 训练GenerRNA  
bash scripts/train_generrna.sh

# 训练MLP
bash scripts/train_mlp.sh

# 通用灵活训练
bash scripts/train_pro2rna.sh
```

## 配置说明

### 数据路径
所有脚本默认使用以下数据路径：
```bash
TRAIN_DATA="/home/yzhang/research/pro2rna/data/output/train.jsonl"
VAL_DATA="/home/yzhang/research/pro2rna/data/output/valid.jsonl"  
TEST_DATA="/home/yzhang/research/pro2rna/data/output/test.jsonl"
```

### 模型配置
- **Taxonomy模型**: `allenai/scibert_scivocab_uncased`
- **Protein模型**: `facebook/esm2_t33_650M_UR50D`
- **融合维度**: 512
- **GPU**: CUDA_VISIBLE_DEVICES=5

### 训练参数对比

| Decoder类型 | Batch Size | Learning Rate | Max mRNA Length | 特殊配置 |
|-------------|------------|---------------|-----------------|----------|
| mRNAGPT     | 4          | 1e-4          | 1024           | LoRA微调 |
| GenerRNA    | 8          | 5e-5          | 256            | 更大词汇表 |
| MLP         | 16         | 1e-3          | 1024           | 从头训练 |

## 输出结果

每次训练完成后，会在输出目录生成：

```
outputs/[decoder_type]_[timestamp]/
├── best_model.pt           # 验证损失最低的模型
├── final_model.pt          # 最终训练完成的模型  
├── tokenizer.json          # 词汇表文件
└── test_results.json       # 测试结果和生成样例
```

## 监控训练

### 1. 终端输出
直接查看训练过程中的损失变化和日志信息

### 2. Weights & Biases
访问wandb项目查看详细的训练曲线：
- mRNAGPT: `flexible-pro2rna-mrna-gpt`
- GenerRNA: `flexible-pro2rna-generrna`  
- MLP: `flexible-pro2rna-mlp`

### 3. 测试结果
```bash
# 查看测试结果
cat outputs/[decoder_type]_[timestamp]/test_results.json | jq .

# 查看生成样例
cat outputs/[decoder_type]_[timestamp]/test_results.json | jq .sample_sequences
```

## 自定义配置

### 修改训练参数
编辑对应的训练脚本，调整以下参数：
- `BATCH_SIZE`: 批次大小
- `LEARNING_RATE`: 学习率
- `EPOCHS`: 训练轮数
- `MAX_MRNA_LENGTH`: 最大mRNA长度

### 修改解码器配置
编辑 `configs/decoder_configs.yaml` 文件，调整：
- 模型架构参数
- LoRA配置
- 生成参数

### 切换GPU
修改脚本中的 `CUDA_VISIBLE_DEVICES` 变量

## 故障排除

### 内存不足
1. 减小批次大小
2. 减小最大序列长度
3. 使用梯度累积

### 训练速度慢
1. 增加批次大小
2. 使用更多GPU
3. 选择MLP decoder进行快速实验

### 模型效果不佳
1. 增加训练轮数
2. 调整学习率
3. 尝试不同的decoder类型
4. 检查数据质量

## 最佳实践

1. **快速实验**: 先用MLP decoder验证数据和代码
2. **生产部署**: 使用mRNAGPT或GenerRNA获得最佳效果
3. **参数调优**: 从配置文件中的默认参数开始
4. **监控训练**: 使用wandb跟踪训练进度
5. **定期保存**: 利用checkpoint机制避免训练中断损失 