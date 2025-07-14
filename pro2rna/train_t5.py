import argparse
from typing import List, Dict
from dataclasses import dataclass, field
import os
import json
import torch
from datasets import Dataset, load_from_disk
from transformers import T5Tokenizer, T5ForConditionalGeneration, TrainingArguments, Trainer, DataCollatorForSeq2Seq

@dataclass
class DataArguments:
    dataset_path: str
    max_length: int = 512

@dataclass
class ModelArguments:
    model_name: str = "t5-base"
    max_length: int = field(default=512)

@dataclass
class TrainingArgs:
    output_dir: str = "./results"
    learning_rate: float = 4e-4
    batch_size: int = 8
    num_train_epochs: int = 3
    warmup_steps: int = 1000

def _resolve_dataset(path: str, split: str):
    """从构建的数据集中加载数据"""
    split_path = os.path.join(path, split)
    if os.path.exists(split_path):
        return load_from_disk(split_path)
    else:
        raise FileNotFoundError(f"Dataset split {split} not found at {split_path}")

def mytok(seq, kmer_len, s):
    """Tokenize a sequence into kmers."""
    seq = seq.upper().replace("T", "U")
    kmer_list = []
    for j in range(0, (len(seq) - kmer_len) + 1, s):
        kmer_list.append(seq[j : j + kmer_len])
    return kmer_list

def convert_to_codon_seq(cds_seq, max_length=512):
    """Convert a DNA sequence to a list of RNA codons and padding."""
    cds_seq = cds_seq.upper().replace("T", "U")
    lst_tok = mytok(cds_seq, 3, 3)
    
    if len(lst_tok) > max_length - 2:
        lst_tok = lst_tok[:max_length - 2] + ['<eos>']
    lst_tok = ['<cls>'] + lst_tok
    lst_tok += ['<pad>'] * (max_length - len(lst_tok))
    
    return lst_tok

def load_and_tokenize_data_from_dataset(dataset, input_tokenizer, label_tokenizer, max_length=512):
    """从数据集对象加载和tokenize数据"""
    input_texts = []
    labels = []
    
    for item in dataset:
        # 构建 taxonomy 信息 - 使用新的字段名称
        taxonomy = (
            f"<{item.get('superkingdom', '')}><{item.get('kingdom', '')}><{item.get('phylum', '')}><{item.get('class', '')}>"
            f"<{item.get('order', '')}><{item.get('family', '')}><{item.get('genus', '')}><{item.get('species', '')}>"
        )
        
        # 构建输入文本（taxonomy + protein_sequence）
        protein_sequence = item["protein_sequence"]
        input_text = f"{taxonomy}{protein_sequence}"
        input_texts.append({"text": input_text})
        
        # 构建标签文本，使用mrna_sequence字段（已经替换了T为U）
        rna_sequence = item.get("mrna_sequence", item.get("cds_sequence", "").replace("T", "U"))
        labels.append({"text": rna_sequence})

    # 对输入文本进行 tokenization
    def tokenize_inputs(examples):
        return input_tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length
        )

    # 对标签文本进行 tokenization
    def tokenize_labels(examples):
        labels = label_tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length
        )["input_ids"]
        return {"labels": labels}

    input_dataset = Dataset.from_list(input_texts)
    label_dataset = Dataset.from_list(labels)

    input_tokenized = input_dataset.map(tokenize_inputs, batched=True, num_proc=6, remove_columns=["text"])
    input_tokenized.set_format("torch", columns=["input_ids", "attention_mask"])

    label_tokenized = label_dataset.map(tokenize_labels, batched=True, num_proc=6, remove_columns=["text"])
    labels_column = label_tokenized["labels"]

    input_tokenized = input_tokenized.add_column("labels", labels_column)
    input_tokenized.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    return input_tokenized

def load_and_tokenize_data_from_jsonl(file_path, input_tokenizer, label_tokenizer, max_length=512):
    """从JSONL文件加载和tokenize数据（向后兼容）"""
    input_texts = []
    labels = []
    
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():
                data = json.loads(line.strip())
                
                # 构建 taxonomy 信息 - 使用新的字段名称
                taxonomy = (
                    f"<{data.get('superkingdom', '')}><{data.get('kingdom', '')}><{data.get('phylum', '')}><{data.get('class', '')}>"
                    f"<{data.get('order', '')}><{data.get('family', '')}><{data.get('genus', '')}><{data.get('species', '')}>"
                )
                
                # 构建输入文本（taxonomy + protein_sequence）
                protein_sequence = data["protein_sequence"]
                input_text = f"{taxonomy}{protein_sequence}"
                input_texts.append({"text": input_text})
                
                # 构建标签文本，使用mrna_sequence字段（已经替换了T为U）
                rna_sequence = data.get("mrna_sequence", data.get("cds_sequence", "").replace("T", "U"))
                labels.append({"text": rna_sequence})

    # 对输入文本进行 tokenization
    def tokenize_inputs(examples):
        return input_tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length
        )

    # 对标签文本进行 tokenization
    def tokenize_labels(examples):
        labels = label_tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length
        )["input_ids"]
        return {"labels": labels}

    input_dataset = Dataset.from_list(input_texts)
    label_dataset = Dataset.from_list(labels)

    input_tokenized = input_dataset.map(tokenize_inputs, batched=True, num_proc=6, remove_columns=["text"])
    input_tokenized.set_format("torch", columns=["input_ids", "attention_mask"])

    label_tokenized = label_dataset.map(tokenize_labels, batched=True, num_proc=6, remove_columns=["text"])
    labels_column = label_tokenized["labels"]

    input_tokenized = input_tokenized.add_column("labels", labels_column)
    input_tokenized.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    return input_tokenized

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser()

    # 更新参数以支持新的数据格式
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the built dataset directory (containing train/valid/test folders).")
    parser.add_argument("--train_data_path", type=str, default=None, help="Path to the training JSONL file (optional, for backward compatibility).")
    parser.add_argument("--valid_data_path", type=str, default=None, help="Path to the validation JSONL file (optional, for backward compatibility).")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length.")
    parser.add_argument("--model_name", type=str, default="t5-base", help="Name of the pretrained T5 model.")
    parser.add_argument("--output_dir", type=str, default="./results", help="Directory to save the model output.")
    parser.add_argument("--learning_rate", type=float, default=4e-4, help="Learning rate for training.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--warmup_steps", type=int, default=1000, help="Number of warmup steps.")
    parser.add_argument("--save_steps", type=int, default=1000, help="Number of steps between each checkpoint save.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Number of gradient accumulation steps.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay rate.")
    parser.add_argument("--save_total_limit", type=int, default=2, help="Total number of checkpoints to keep.")
    parser.add_argument("--fp16", action="store_true", help="Enable FP16 training.")
    parser.add_argument("--logging_dir", type=str, default="./logs", help="Directory for training logs.")
    parser.add_argument("--logging_steps", type=int, default=200, help="Logging frequency in steps.")
    parser.add_argument("--eval_strategy", type=str, default="epoch", help="Evaluation strategy (e.g., 'epoch', 'steps').")
    parser.add_argument("--prediction_loss_only", type=bool, default=True, help="Whether to compute only prediction loss during evaluation.")
    parser.add_argument("--report_to", type=str, default="wandb", help="Reporting tool (e.g., 'wandb', 'tensorboard').")
    parser.add_argument("--run_name", type=str, default="CodonT5", help="specify wandb run name.")

    args = parser.parse_args()

    model_args = ModelArguments(model_name=args.model_name)
    train_args = TrainingArgs(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        num_train_epochs=args.num_train_epochs,
        warmup_steps=args.warmup_steps
    )

    # 加载tokenizer
    input_tokenizer = T5Tokenizer.from_pretrained("ZYMScott/CodonT5_input")
    label_tokenizer = T5Tokenizer.from_pretrained("ZYMScott/CodonT5_label")

    print('input_tokenizer vocabulary size: ', len(input_tokenizer))
    print('label_tokenizer vocabulary size: ', len(label_tokenizer))

    # 初始化模型
    model = T5ForConditionalGeneration.from_pretrained(model_args.model_name).to(device)
    model.resize_token_embeddings(len(input_tokenizer))
    model.lm_head = torch.nn.Linear(
        model.config.d_model,
        len(label_tokenizer),
        bias=False
    ).to(device) 
    
    # Calculate model parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total model parameters: {total_params / 1e6:.2f}M")
    
    print(f'Model: {model}')

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=input_tokenizer,
        model=model,
        padding=True
    )

    # 加载数据 - 支持新格式和旧格式
    if args.train_data_path and args.valid_data_path:
        # 向后兼容：从JSONL文件加载
        print("Loading data from JSONL files...")
        train_dataset = load_and_tokenize_data_from_jsonl(
            file_path=args.train_data_path,
            input_tokenizer=input_tokenizer,
            label_tokenizer=label_tokenizer,
            max_length=args.max_length,
        )

        valid_dataset = load_and_tokenize_data_from_jsonl(
            file_path=args.valid_data_path,
            input_tokenizer=input_tokenizer,
            label_tokenizer=label_tokenizer,
            max_length=args.max_length,
        )
    else:
        # 新格式：从构建的数据集加载
        print("Loading data from built dataset...")
        train_raw_dataset = _resolve_dataset(args.dataset_path, "train")
        valid_raw_dataset = _resolve_dataset(args.dataset_path, "valid")
        
        print(f"Loaded {len(train_raw_dataset)} training samples")
        print(f"Loaded {len(valid_raw_dataset)} validation samples")
        
        train_dataset = load_and_tokenize_data_from_dataset(
            dataset=train_raw_dataset,
            input_tokenizer=input_tokenizer,
            label_tokenizer=label_tokenizer,
            max_length=args.max_length,
        )

        valid_dataset = load_and_tokenize_data_from_dataset(
            dataset=valid_raw_dataset,
            input_tokenizer=input_tokenizer,
            label_tokenizer=label_tokenizer,
            max_length=args.max_length,
        )

    print(f"Processed {len(train_dataset)} training samples")
    print(f"Processed {len(valid_dataset)} validation samples")

    training_args = TrainingArguments(
        output_dir=train_args.output_dir,
        overwrite_output_dir=True,
        per_device_train_batch_size=train_args.batch_size,
        num_train_epochs=train_args.num_train_epochs,
        save_steps=args.save_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        fp16=args.fp16,
        warmup_steps=train_args.warmup_steps,
        save_total_limit=args.save_total_limit,
        logging_dir=args.logging_dir,
        logging_steps=args.logging_steps,
        evaluation_strategy=args.eval_strategy,
        run_name=args.run_name,
        save_safetensors=False,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        report_to=args.report_to,
    )

    # 检查是否有checkpoint可以恢复
    checkpoint_path = None
    if os.path.exists(args.output_dir):
        checkpoints = [f.path for f in os.scandir(args.output_dir) if f.is_dir() and f.name.startswith('checkpoint-')]
        if checkpoints:
            checkpoint_path = max(checkpoints)  # 选择最新的checkpoint
            print(f"Resuming training from checkpoint: {checkpoint_path}")
    else: 
        print("Training from scratch")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset, 
        data_collator=data_collator,
    )

    if checkpoint_path:
        trainer.train(resume_from_checkpoint=checkpoint_path)
    else:
        trainer.train()

    # 保存最终模型
    trainer.save_model()
    
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()

if __name__ == "__main__":
    main()
