import argparse
import torch
from torch.utils.data import DataLoader
from datasets import Dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration
import json
from tqdm import tqdm


def load_and_tokenize_data(file_path, tokenizer, max_length=512):
    input_texts = []
    labels = []

    with open(file_path, 'r') as f:
        data_list = json.load(f)

    for data in data_list:
        taxonomy = (
            f"<{data['Kindom']}><{data['Phylum']}><{data['Class']}>"
            f"<{data['Order']}><{data['Family']}><{data['Genus']}><{data['Species']}>"
        )

        prot_sequence = data["prot_sequence"]
        input_text = f"{taxonomy}{prot_sequence}"
        input_texts.append({"text": input_text})

        rna_sequence = data["cds_sequence"].replace("T", "U")
        labels.append({"text": rna_sequence})

    def tokenize_function(examples):
        model_inputs = tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length
        )
        return model_inputs

    input_dataset = Dataset.from_list(input_texts)
    label_dataset = Dataset.from_list(labels)

    input_tokenized = input_dataset.map(tokenize_function, batched=True, num_proc=6, remove_columns=["text"])
    input_tokenized.set_format("torch", columns=["input_ids", "attention_mask"])

    label_tokenized = label_dataset.map(tokenize_function, batched=True, num_proc=6, remove_columns=["text"])
    label_tokenized.set_format("torch", columns=["input_ids"])

    true_labels_tensor = torch.stack([torch.tensor(x["input_ids"]).clone().detach() for x in label_tokenized])
    return input_tokenized, true_labels_tensor, labels, label_tokenized

def decode_logits_to_text(logits, tokenizer):
    """
    decode until first stop sign </s>
    """
    predicted_ids = torch.argmax(logits, dim=-1)
    # 截断到第一个 </s> (id=1) 出现的位置
    truncated_ids = []
    for sequence in predicted_ids:
        truncated_sequence = []
        for token_id in sequence.tolist():
            truncated_sequence.append(token_id)
            if token_id == 1:  # </s> token ID
                break
        truncated_ids.append(truncated_sequence)

    decoded_text = tokenizer.batch_decode(
        truncated_ids, skip_special_tokens=True
    )

    cleaned_text = [text.replace(" ", "").replace("\n", "") for text in decoded_text]
    return cleaned_text


def decode_logits_to_text(logits, tokenizer, input_ids, pad_token_id=0, start_token_id=329):
    """
    Decode logits to text based on input_ids, using the range from the first occurrence of start_token_id (329)
    to the first occurrence of pad_token_id (0). If pad_token_id is not present, decode to the end.
    """
    predicted_ids = torch.argmax(logits, dim=-1)
    truncated_ids = []
    
    for i, sequence in enumerate(predicted_ids):
        input_seq = input_ids[i].tolist()
        
        if start_token_id in input_seq:
            start_index = input_seq.index(start_token_id)
        else:
            raise ValueError(f"Start token ID {start_token_id} not found in input_ids[{i}]")
        
        if pad_token_id in input_seq:
            end_index = input_seq.index(pad_token_id)
        else:
            end_index = len(input_seq)
        
        # 计算有效长度
        valid_length = max(0, end_index - start_index) 
        
        truncated_sequence = sequence[:valid_length].tolist()
        truncated_ids.append(truncated_sequence)

    decoded_text = tokenizer.batch_decode(
        truncated_ids, skip_special_tokens=False
    )
    cleaned_text = [text.replace(" ", "").replace("\n", "") for text in decoded_text]
    return cleaned_text

def decode_logits_to_text_label(logits, tokenizer, label_ids):
    """
    Decode logits to text with length matching label_ids. The length is determined by label_ids from
    the start to the first occurrence of token ID 1 (</s>). If no token ID 1 is present, decode to the end.
    """
    predicted_ids = torch.argmax(logits, dim=-1)
    truncated_ids = []
    
    for i, sequence in enumerate(predicted_ids):
        label_seq = label_ids[i].tolist()

        # 找到第一个 </s> (token ID=1) 的位置
        if 1 in label_seq:
            valid_length = label_seq.index(1)
        else:
            valid_length = len(label_seq)
        
        truncated_sequence = sequence[:valid_length].tolist()
        truncated_ids.append(truncated_sequence)

    decoded_text = tokenizer.batch_decode(
        truncated_ids, skip_special_tokens=False
    )
    cleaned_text = [text.replace(" ", "").replace("\n", "") for text in decoded_text]
    return cleaned_text

def calculate_token_level_accuracy(predictions, labels):
    total_tokens = 0
    matched_tokens = 0

    for pred, label in zip(predictions, labels):
        pred_tokens = pred["input_ids"]
        true_tokens = label["input_ids"]

        pred_tokens = [t for t in pred_tokens if t != 0]
        true_tokens = [t for t in true_tokens if t != 0]


        min_length = min(len(pred_tokens), len(true_tokens))
        total_tokens += min_length

        for i in range(min_length):
            if pred_tokens[i] == true_tokens[i]:
                matched_tokens += 1

    token_accuracy = matched_tokens / total_tokens if total_tokens > 0 else 0
    return token_accuracy

def evaluate(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load tokenizer and model
    tokenizer = T5Tokenizer.from_pretrained("ZYMScott/CodonT5")
    model = T5ForConditionalGeneration.from_pretrained(args.model_name).to(device)

    # Load test data
    test_dataset, true_labels, true_texts, true_labels_tokenized = load_and_tokenize_data(
        file_path=args.test_data_path,
        tokenizer=tokenizer,
        max_length=args.max_length,
    )

    # Create DataLoader for batching
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model.eval()
    predictions = []


    with open(args.output_file, 'w') as output_file:
        output_file.write("True RNA Sequence,Predicted RNA Sequence\n")

        for batch_idx, batch in enumerate(tqdm(test_dataloader)):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            start_idx = batch_idx * args.batch_size
            end_idx = start_idx + input_ids.size(0)
            true_label_batch = true_labels[start_idx:end_idx].to(device)
            # print("Shape of true_labels:", true_label_batch.shape)

            # print('true_labels:[:10]', true_label_batch[0])
            # print('true_labels:[-10:]', true_label_batch[0, -10:])

            # label_ids = true_labels_tokenized["input_ids"][start_idx: end_idx].to(device)
            # print('shape: ', label_ids.shape)
            # print('label_ids:[:10]', label_ids[0, :10])
            # print('label_ids:[-10:]', label_ids[0, -10:])

            # Create decoder input IDs
            decoder_input_ids = torch.cat([
                torch.full((input_ids.size(0), 1), tokenizer.pad_token_id, dtype=torch.long, device=device), 
                true_label_batch[:, :-1].to(device)  # Shifted labels
            ], dim=1)
            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids)
                logits = outputs.logits


            # pred_texts = decode_logits_to_text(logits, tokenizer)
            # pred_texts = decode_logits_to_text(logits, tokenizer, input_ids)
            pred_texts = decode_logits_to_text_label(logits, tokenizer, true_label_batch)



            for i, pred_text in enumerate(pred_texts):
                # true_text = true_texts[i]["text"].replace(" ", "").replace("\n", "")
                true_text = true_texts[start_idx + i]["text"].replace(" ", "").replace("\n", "")

                # print('true_texts:[:10]', true_texts[start_idx + i]["text"])
                # print('true_texts:[-10:]', true_texts[start_idx + i]["text"])

                pred_tokens = tokenizer(pred_text, return_tensors="pt", truncation=True, max_length=args.max_length)["input_ids"].squeeze().tolist()
                true_tokens = true_labels[i].squeeze().tolist()

                # print(f"Batch {batch_idx}, Sample {i}:")
                # print(f"  Predicted Length: {len(pred_text)}")
                # print(f"  True Length: {len(true_text)}")

                predictions.append({"input_ids": pred_tokens})
                output_file.write(f"{true_text},{pred_text}\n")

    token_accuracy = calculate_token_level_accuracy(predictions, true_labels_tokenized)
    print(f"Token-level accuracy: {token_accuracy:.4f}")
    print("Evaluation completed and saved to", args.output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_data_path", type=str, required=True, help="Path to the test data.")
    parser.add_argument("--model_name", type=str, default="t5-base", help="Name or path of the pretrained T5 model.")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length for the model.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for evaluation.")
    parser.add_argument("--output_file", type=str, default="predictions_forward_length_label.csv", help="File to save true and predicted sequences.")
    args = parser.parse_args()

    evaluate(args)

# CUDA_VISIBLE_DEVICES=1 python evaluate_codont5_forward.py --test_data_path /raid_elmo/home/lr/zym/protein2rna/ncbi_dataset/data/test_flat.json --model_name /raid_elmo/home/lr/zym/protein2rna/checkpoints/codont5/checkpoint-1413000 --batch_size 128