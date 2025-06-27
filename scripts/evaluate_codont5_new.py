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
    return input_tokenized, true_labels_tensor, label_tokenized

def decode_logits_and_labels(logits, tokenizer, label_ids):
    """
    Decode logits and label_ids to text with length determined by label_ids.
    The length is determined by label_ids from the start to the first occurrence
    of token ID 1 (</s>). If no token ID 1 is present, decode to the end.
    """
    predicted_ids = torch.argmax(logits, dim=-1)
    truncated_predicted_ids = []
    truncated_label_ids = []
    total_tokens = 0
    matched_tokens = 0

    for i, sequence in enumerate(predicted_ids):
        label_seq = label_ids[i].tolist()

        # 找到第一个 </s> (token ID=1) 的位置
        if 1 in label_seq:
            valid_length = label_seq.index(1)
        else:
            valid_length = len(label_seq)

        # 截断 predicted_ids 和 label_ids 到相同的 valid_length
        truncated_predicted_sequence = sequence[:valid_length].tolist()
        truncated_label_sequence = label_seq[:valid_length]

        truncated_predicted_ids.append(truncated_predicted_sequence)
        truncated_label_ids.append(truncated_label_sequence)

        # 计算截断后的 token level accuracy
        total_tokens += len(truncated_label_sequence)
        matched_tokens += sum(p == t for p, t in zip(truncated_predicted_sequence, truncated_label_sequence))

    decoded_predicted_text = tokenizer.batch_decode(
        truncated_predicted_ids, skip_special_tokens=True
    )
    decoded_label_text = tokenizer.batch_decode(
        truncated_label_ids, skip_special_tokens=True
    )
    cleaned_predicted_text = [text.replace(" ", "").replace("\n", "") for text in decoded_predicted_text]
    cleaned_label_text = [text.replace(" ", "").replace("\n", "") for text in decoded_label_text]

    token_level_accuracy = matched_tokens / total_tokens if total_tokens > 0 else 0

    return predicted_ids, cleaned_predicted_text, cleaned_label_text, token_level_accuracy


def calculate_token_level_accuracy(predictions, labels):
    total_tokens = 0
    matched_tokens = 0

    for pred_tokens, true_tokens in zip(predictions, labels):
        pred_tokens = [t for t in pred_tokens.tolist() if t != 0]
        true_tokens = [t for t in true_tokens.tolist() if t != 0]

    token_accuracy = matched_tokens / total_tokens if total_tokens > 0 else 0
    return token_accuracy



def evaluate(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load tokenizer and model
    tokenizer = T5Tokenizer.from_pretrained("ZYMScott/CodonT5")
    model = T5ForConditionalGeneration.from_pretrained(args.model_name).to(device)

    # Load test data
    test_dataset, true_labels, label_tokenized = load_and_tokenize_data(
        file_path=args.test_data_path,
        tokenizer=tokenizer,
        max_length=args.max_length,
    )

    # Create DataLoader for batching
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model.eval()

    with open(args.output_file, 'w') as output_file:
        output_file.write("True RNA Sequence,Predicted RNA Sequence\n")

        all_logits = []
        all_label_ids = []
        predictions = []

        total_matched_tokens = 0
        total_tokens = 0
        for batch_idx, batch in enumerate(tqdm(test_dataloader)):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            start_idx = batch_idx * args.batch_size
            end_idx = start_idx + input_ids.size(0)
            true_label_batch = true_labels[start_idx:end_idx].to(device)

            # Create decoder input IDs
            decoder_input_ids = torch.cat([
                torch.full((input_ids.size(0), 1), tokenizer.pad_token_id, dtype=torch.long, device=device),
                true_label_batch[:, :-1].to(device)  # Shifted labels
            ], dim=1)

            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids)
                logits = outputs.logits

            pred_ids, pred_texts, true_texts_decoded, batch_token_accuracy = decode_logits_and_labels(logits, tokenizer, true_label_batch)

            # predictions.append(pred_ids)
            total_matched_tokens += batch_token_accuracy * len(true_label_batch)
            total_tokens += len(true_label_batch)

            for pred_text, true_text in zip(pred_texts, true_texts_decoded):
                truncated_true_text = true_text[:len(pred_text)]
                output_file.write(f"{truncated_true_text},{pred_text}\n")

                
        # Flatten all logits and labels for accuracy calculation
        token_accuracy = total_matched_tokens / total_tokens if total_tokens > 0 else 0
        # token_accuracy = calculate_token_level_accuracy(predictions, true_labels)
        print(f"Token-level accuracy: {token_accuracy:.4f}")
        print("Evaluation completed and saved to", args.output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_data_path", type=str, required=True, help="Path to the test data.")
    parser.add_argument("--model_name", type=str, default="t5-base", help="Name or path of the pretrained T5 model.")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length for the model.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for evaluation.")
    parser.add_argument("--output_file", type=str, default="predictions_length_label.csv", help="File to save true and predicted sequences.")
    args = parser.parse_args()

    evaluate(args)

# CUDA_VISIBLE_DEVICES=1 python evaluate_codont5_forward.py --test_data_path /raid_elmo/home/lr/zym/protein2rna/ncbi_dataset/data/test_flat.json --model_name /raid_elmo/home/lr/zym/protein2rna/checkpoints/codont5/checkpoint-1413000 --batch_size 128