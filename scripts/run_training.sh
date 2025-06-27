#!/bin/bash

# FlexiblePro2RNA 训练脚本管理器
# 提供交互式选择不同decoder类型进行训练

set -e

PROJECT_ROOT="/home/yzhang/research/pro2rna"
cd "$PROJECT_ROOT"

echo "========================================"
echo "    FlexiblePro2RNA 训练脚本管理器"
echo "========================================"
echo ""
echo "可用的训练选项："
echo "1. mRNAGPT Decoder (codon级别, vocab=69, 预训练模型)"
echo "2. GenerRNA Decoder (k-mer级别, vocab=1024, 预训练模型)"
echo "3. MLP Decoder (codon级别, vocab=69, 轻量级从头训练)"
echo "4. 通用灵活训练 (可自定义decoder类型)"
echo "5. 显示训练脚本内容"
echo "6. 退出"
echo ""

while true; do
    read -p "请选择训练选项 (1-6): " choice
    
    case $choice in
        1)
            echo ""
            echo "启动 mRNAGPT Decoder 训练..."
            echo "特点："
            echo "- 基于mRNAdesigner预训练模型"
            echo "- Codon级别tokenization (词汇表69)"
            echo "- LoRA微调，较小内存需求"
            echo "- 适合生成高质量mRNA序列"
            echo ""
            read -p "确认启动? (y/N): " confirm
            if [[ $confirm =~ ^[Yy]$ ]]; then
                bash scripts/train_mrna_gpt.sh
            fi
            break
            ;;
        2)
            echo ""
            echo "启动 GenerRNA Decoder 训练..."
            echo "特点："
            echo "- 基于GenerRNA预训练模型"
            echo "- K-mer级别tokenization (词汇表1024)"
            echo "- 支持更复杂的序列模式"
            echo "- 适合核苷酸级别的精确控制"
            echo ""
            read -p "确认启动? (y/N): " confirm
            if [[ $confirm =~ ^[Yy]$ ]]; then
                bash scripts/train_generrna.sh
            fi
            break
            ;;
        3)
            echo ""
            echo "启动 MLP Decoder 训练..."
            echo "特点："
            echo "- 轻量级全连接网络"
            echo "- 从头训练，快速迭代"
            echo "- Codon级别tokenization (词汇表69)"
            echo "- 适合快速原型和基线对比"
            echo ""
            read -p "确认启动? (y/N): " confirm
            if [[ $confirm =~ ^[Yy]$ ]]; then
                bash scripts/train_mlp.sh
            fi
            break
            ;;
        4)
            echo ""
            echo "启动通用灵活训练..."
            echo "可以通过修改脚本参数自定义decoder类型"
            echo ""
            read -p "确认启动? (y/N): " confirm
            if [[ $confirm =~ ^[Yy]$ ]]; then
                bash scripts/train_pro2rna.sh
            fi
            break
            ;;
        5)
            echo ""
            echo "可用的训练脚本："
            echo "- scripts/train_mrna_gpt.sh (mRNAGPT专用)"
            echo "- scripts/train_generrna.sh (GenerRNA专用)"
            echo "- scripts/train_mlp.sh (MLP专用)"
            echo "- scripts/train_pro2rna.sh (通用灵活)"
            echo ""
            read -p "查看哪个脚本的内容? (1-4, 或按Enter返回): " script_choice
            case $script_choice in
                1) echo ""; cat scripts/train_mrna_gpt.sh ;;
                2) echo ""; cat scripts/train_generrna.sh ;;
                3) echo ""; cat scripts/train_mlp.sh ;;
                4) echo ""; cat scripts/train_pro2rna.sh ;;
                *) echo "返回主菜单..." ;;
            esac
            echo ""
            ;;
        6)
            echo "退出训练脚本管理器"
            exit 0
            ;;
        *)
            echo "无效选择，请输入 1-6"
            ;;
    esac
done

echo ""
echo "训练启动完成！"
echo ""
echo "监控训练进度："
echo "1. 查看终端输出"
echo "2. 使用 wandb 查看训练曲线"
echo "3. 检查输出目录中的日志文件"
echo ""
echo "训练完成后，可在输出目录找到："
echo "- best_model.pt (最优模型)"
echo "- final_model.pt (最终模型)"
echo "- tokenizer.json (词汇表)"
echo "- test_results.json (测试结果)" 