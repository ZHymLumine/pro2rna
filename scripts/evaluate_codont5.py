import argparse
import torch
from torch.utils.data import DataLoader
from datasets import load_from_disk
from transformers import T5Tokenizer, T5ForConditionalGeneration, DataCollatorForSeq2Seq
from tqdm import tqdm
import os
import numpy as np


def load_and_tokenize_data_from_dataset(dataset, input_tokenizer, label_tokenizer, max_length=512):
    """从构建的数据集加载和tokenize数据（与新训练代码一致）"""
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
        # Set padding tokens to -100 (ignored in loss calculation)
        pad_token_id = label_tokenizer.pad_token_id
        vocab_size = len(label_tokenizer)
        # Validate and fix token IDs
        validated_labels = []
        for label in labels:
            validated_label = []
            for token_id in label:
                if token_id == pad_token_id:
                    validated_label.append(-100)
                elif token_id >= vocab_size:
                    # Clamp to valid range
                    validated_label.append(vocab_size - 1)
                elif token_id < 0:
                    validated_label.append(-100)
                else:
                    validated_label.append(token_id)
            validated_labels.append(validated_label)
        return {"labels": validated_labels}

    from datasets import Dataset
    input_dataset = Dataset.from_list(input_texts)
    label_dataset = Dataset.from_list(labels)

    input_tokenized = input_dataset.map(tokenize_inputs, batched=True, num_proc=6, remove_columns=["text"])
    input_tokenized.set_format("torch", columns=["input_ids", "attention_mask"])

    label_tokenized = label_dataset.map(tokenize_labels, batched=True, num_proc=6, remove_columns=["text"])
    labels_column = label_tokenized["labels"]

    input_tokenized = input_tokenized.add_column("labels", labels_column)
    input_tokenized.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    return input_tokenized


def decode_logits_and_labels(logits, label_tokenizer, label_ids):
    """Decode logits and label_ids to text with length determined by label_ids"""
    predicted_ids = torch.argmax(logits, dim=-1)
    truncated_predicted_ids = []
    truncated_label_ids = []

    for i, sequence in enumerate(predicted_ids):
        label_seq = label_ids[i].tolist()

        # Find first </s> (token ID=1) position, or use -100 as end marker
        valid_length = len(label_seq)
        for j, token_id in enumerate(label_seq):
            if token_id == -100:  # Padding token (ignored in loss)
                valid_length = j
                break
            if token_id == 1:  # </s> token
                valid_length = j
                break

        truncated_predicted_sequence = sequence[:valid_length].tolist()
        truncated_label_sequence = label_seq[:valid_length]

        truncated_predicted_ids.append(truncated_predicted_sequence)
        truncated_label_ids.append(truncated_label_sequence)

    decoded_predicted_text = label_tokenizer.batch_decode(
        truncated_predicted_ids, skip_special_tokens=False
    )
    decoded_label_text = label_tokenizer.batch_decode(
        truncated_label_ids, skip_special_tokens=False
    )
    cleaned_predicted_text = [text.replace(" ", "").replace("\n", "") for text in decoded_predicted_text]
    cleaned_label_text = [text.replace(" ", "").replace("\n", "") for text in decoded_label_text]

    return predicted_ids, cleaned_predicted_text, cleaned_label_text


def calculate_token_level_accuracy_fast(all_preds, all_labels, pad_token_id):
    """
    Efficiently calculate token-level accuracy, excluding pad tokens.
    """
    preds_concat = np.concatenate(all_preds, axis=0)
    labels_concat = np.concatenate(all_labels, axis=0)

    # Create mask for non-padding and non-ignored tokens
    non_pad_mask = (labels_concat != pad_token_id) & (labels_concat != -100)

    total_tokens = np.sum(non_pad_mask)
    matched_tokens = np.sum((preds_concat == labels_concat) & non_pad_mask)

    token_accuracy = matched_tokens / total_tokens if total_tokens > 0 else 0
    return token_accuracy


def evaluate(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load input and label tokenizers
    input_tokenizer = T5Tokenizer.from_pretrained("ZYMScott/CodonT5_input")
    label_tokenizer = T5Tokenizer.from_pretrained("ZYMScott/CodonT5_label")

    # Load model
    model = T5ForConditionalGeneration.from_pretrained(args.model_name).to(device)
    model.resize_token_embeddings(len(input_tokenizer))
    
    # Load checkpoint first to get the correct lm_head size
    checkpoint_path = args.checkpoint if args.checkpoint else os.path.join(args.model_name, "pytorch_model.bin")
    
    if os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location=device)
        # Handle both full state_dict and model-only state_dict
        if state_dict and 'model.' in list(state_dict.keys())[0]:
            # Remove 'model.' prefix if present
            state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
        
        # Get lm_head vocab size from checkpoint
        if 'lm_head.weight' in state_dict:
            checkpoint_vocab_size = state_dict['lm_head.weight'].shape[0]
            print(f"Checkpoint lm_head vocab size: {checkpoint_vocab_size}")
            print(f"Current label_tokenizer vocab size: {len(label_tokenizer)}")
            
            # Create lm_head with checkpoint vocab size and load weights
            new_lm_head = torch.nn.Linear(
                model.config.d_model,
                checkpoint_vocab_size,
                bias=False
            ).to(device)
            new_lm_head.weight.data = state_dict['lm_head.weight']
            
            # Assign lm_head before loading other weights
            model.lm_head = new_lm_head
            
            # Remove lm_head from state_dict to avoid shape mismatch
            state_dict_without_lm_head = {k: v for k, v in state_dict.items() if k != 'lm_head.weight'}
            
            # Load checkpoint weights (excluding lm_head which we already loaded)
            model.load_state_dict(state_dict_without_lm_head, strict=False)
            
            print(f"Loaded checkpoint from {checkpoint_path}")
            print(f"Using checkpoint lm_head vocab size: {checkpoint_vocab_size}")
        else:
            # Fallback: use label_tokenizer vocab size
            print("Warning: lm_head.weight not found in checkpoint, using label_tokenizer vocab size")
            new_lm_head = torch.nn.Linear(
                model.config.d_model,
                len(label_tokenizer),
                bias=False
            ).to(device)
            model.load_state_dict(state_dict, strict=False)
            model.lm_head = new_lm_head
    else:
        # No checkpoint: create new lm_head with label_tokenizer vocab size
        print(f"Warning: Checkpoint not found at {checkpoint_path}, initializing new lm_head")
        new_lm_head = torch.nn.Linear(
            model.config.d_model,
            len(label_tokenizer),
            bias=False
        ).to(device)
        model.lm_head = new_lm_head
    
    print(f"Model loaded: {args.model_name}")
    print(f"Input tokenizer vocab size: {len(input_tokenizer)}")
    print(f"Label tokenizer vocab size: {len(label_tokenizer)}")
    print(f"Label tokenizer pad_token_id: {label_tokenizer.pad_token_id}")
    print(f"Model decoder_start_token_id: {model.config.decoder_start_token_id if hasattr(model.config, 'decoder_start_token_id') else 'Not set (default: 0)'}")
    print(f"Max length: {args.max_length}")

    # Load test dataset
    if os.path.isdir(args.test_data_path):
        # New format: dataset directory
        test_raw_dataset = load_from_disk(os.path.join(args.test_data_path, "test"))
        print(f"Loaded {len(test_raw_dataset)} test samples from dataset directory")
        
        test_dataset = load_and_tokenize_data_from_dataset(
            dataset=test_raw_dataset,
            input_tokenizer=input_tokenizer,
            label_tokenizer=label_tokenizer,
            max_length=args.max_length,
        )
    else:
        raise ValueError(f"Test data path must be a directory containing 'test' subdirectory. Got: {args.test_data_path}")

    # Create DataLoader
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=input_tokenizer,
        model=model,
        padding=True
    )
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=data_collator)

    model.eval()

    with open(args.output_file, 'w') as output_file:
        output_file.write("True RNA Sequence,Predicted RNA Sequence\n")

        all_preds = []
        all_labels = []
        
        for batch_idx, batch in enumerate(tqdm(test_dataloader)):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            # Validate token IDs are within vocab range
            vocab_size = len(label_tokenizer)
            invalid_mask = (labels >= vocab_size) & (labels != -100)
            if torch.any(invalid_mask):
                invalid_count = torch.sum(invalid_mask).item()
                max_invalid = torch.max(labels[invalid_mask]).item()
                print(f"Warning: Found {invalid_count} invalid token IDs in batch {batch_idx} (max: {max_invalid}, vocab_size: {vocab_size})")
                # Clamp invalid token IDs to valid range
                labels = torch.where(invalid_mask, torch.tensor(vocab_size - 1, device=device), labels)
            
            # Create decoder input IDs (shifted labels)
            # Replace -100 with pad_token_id for decoder input
            labels_for_decoder = labels.clone()
            labels_for_decoder[labels_for_decoder == -100] = label_tokenizer.pad_token_id
            
            # Validate pad_token_id is within vocab range
            if label_tokenizer.pad_token_id >= vocab_size:
                print(f"Error: pad_token_id ({label_tokenizer.pad_token_id}) >= vocab_size ({vocab_size})")
                labels_for_decoder[labels_for_decoder == label_tokenizer.pad_token_id] = 0  # Use 0 as fallback
            
            # Use decoder_start_token_id (usually 0 for T5)
            decoder_start_token_id = model.config.decoder_start_token_id if hasattr(model.config, 'decoder_start_token_id') else 0
            if decoder_start_token_id >= vocab_size:
                decoder_start_token_id = 0
                print(f"Warning: decoder_start_token_id >= vocab_size, using 0 instead")
            
            decoder_input_ids = torch.cat([
                torch.full((input_ids.size(0), 1), decoder_start_token_id, dtype=torch.long, device=device),
                labels_for_decoder[:, :-1]
            ], dim=1)
            
            # Ensure decoder_input_ids doesn't exceed max_length
            original_length = decoder_input_ids.size(1)
            if decoder_input_ids.size(1) > args.max_length:
                decoder_input_ids = decoder_input_ids[:, :args.max_length]
                print(f"Warning: Truncated decoder_input_ids from {original_length} to {args.max_length}")
            
            # Final validation: ensure all decoder_input_ids are within vocab range
            if torch.any(decoder_input_ids >= vocab_size) or torch.any(decoder_input_ids < 0):
                decoder_input_ids = torch.clamp(decoder_input_ids, min=0, max=vocab_size - 1)
                print(f"Warning: Clamped decoder_input_ids to valid range [0, {vocab_size - 1}]")
            
            # Create decoder attention mask (1 for non-padding tokens, 0 for padding)
            decoder_attention_mask = (decoder_input_ids != label_tokenizer.pad_token_id).long()
            # If pad_token_id is invalid, use decoder_start_token_id as reference
            if label_tokenizer.pad_token_id >= vocab_size:
                decoder_attention_mask = (decoder_input_ids != decoder_start_token_id).long()

            with torch.no_grad():
                try:
                    outputs = model(
                        input_ids=input_ids, 
                        attention_mask=attention_mask, 
                        decoder_input_ids=decoder_input_ids,
                        decoder_attention_mask=decoder_attention_mask
                    )
                    logits = outputs.logits
                except RuntimeError as e:
                    if "CUDA" in str(e) or "assert" in str(e).lower():
                        print(f"CUDA error in batch {batch_idx}: {e}")
                        print(f"Input shape: {input_ids.shape}, Decoder input shape: {decoder_input_ids.shape}")
                        print(f"Input IDs range: [{input_ids.min().item()}, {input_ids.max().item()}]")
                        print(f"Decoder IDs range: [{decoder_input_ids.min().item()}, {decoder_input_ids.max().item()}]")
                        print(f"Vocab size: {vocab_size}")
                        raise
                    else:
                        raise

            # Decode logits and calculate accuracy
            pred_ids, pred_texts, true_texts_decoded = decode_logits_and_labels(
                logits, label_tokenizer, labels
            )

            pred_ids = pred_ids.detach().cpu().numpy()
            labels_np = labels.detach().cpu().numpy()

            all_preds.append(pred_ids)
            all_labels.append(labels_np)

            for pred_text, true_text in zip(pred_texts, true_texts_decoded):
                output_file.write(f"{true_text},{pred_text}\n")

        token_accuracy = calculate_token_level_accuracy_fast(all_preds, all_labels, label_tokenizer.pad_token_id)
        print(f"Token-level accuracy: {token_accuracy:.4f}")
        print("Evaluation completed and saved to", args.output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_data_path", type=str, required=True, 
                        help="Path to the test dataset directory (containing 'test' subdirectory) or built dataset path.")
    parser.add_argument("--model_name", type=str, default="t5-base", 
                        help="Name or path of the pretrained T5 model.")
    parser.add_argument("--checkpoint", type=str, default=None, 
                        help="Path to the model checkpoint (pytorch_model.bin). If None, will try to load from model_name/pytorch_model.bin")
    parser.add_argument("--max_length", type=int, default=512, 
                        help="Maximum sequence length for the model.")
    parser.add_argument("--batch_size", type=int, default=8, 
                        help="Batch size for evaluation.")
    parser.add_argument("--output_file", type=str, default="predictions_new_format.csv", 
                        help="File to save true and predicted sequences.")
    args = parser.parse_args()

    evaluate(args)


# CUDA_VISIBLE_DEVICES=3 python scripts/evaluate_codont5.py --test_data_path /home/yzhang/research/pro2rna/data/build --model_name t5-base --checkpoint /home/yzhang/research/pro2rna/checkpoints/codont5_new_format/pytorch_model.bin --batch_size 16  --output_file predictions.csv