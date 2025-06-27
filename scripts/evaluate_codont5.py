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
    labels_column = label_tokenized["input_ids"]
    input_tokenized = input_tokenized.add_column("labels", labels_column)
    input_tokenized.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    return input_tokenized

# def calculate_token_level_accuracy(predictions, labels):
#     total_tokens = 0
#     matched_tokens = 0

#     for pred, true in zip(predictions, labels):
#         pred_tokens = pred["input_ids"]
#         true_tokens = true["input_ids"]

#         pred_tokens = [t for t in pred_tokens if t != -100]
#         true_tokens = [t for t in true_tokens if t != -100]

#         min_length = min(len(pred_tokens), len(true_tokens))
#         total_tokens += min_length

#         for i in range(min_length):
#             if pred_tokens[i] == true_tokens[i]:
#                 matched_tokens += 1

#     token_accuracy = matched_tokens / total_tokens if total_tokens > 0 else 0
#     return token_accuracy

def calculate_token_level_accuracy(predictions, labels):
    """
    计算 token-level accuracy。
    """
    total_tokens = 0
    matched_tokens = 0

    for pred_tokens, true_tokens in zip(predictions, labels):
        pred_tokens = [t for t in pred_tokens if t != 0]
        true_tokens = [t for t in true_tokens if t != 0]

        min_length = min(len(pred_tokens), len(true_tokens))
        total_tokens += min_length

        for i in range(min_length):
            if pred_tokens[i] == true_tokens[i]:
                matched_tokens += 1

    token_accuracy = matched_tokens / total_tokens if total_tokens > 0 else 0
    return token_accuracy


def generate_consistent_length(model, tokenizer, input_ids, attention_mask, true_length, max_length, device, max_attempts=10):
    """
    生成与目标序列长度一致的预测序列，最多尝试 max_attempts 次。
    """
    attempts = 0
    closest_pred_tokens = None
    closest_pred_text = None
    closest_length_diff = float("inf")

    while attempts < max_attempts:
        attempts += 1
        with torch.no_grad():
            pred_tokens = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_length,
                num_return_sequences=1,
                do_sample=True,
                temperature=0.7,
                top_k=50,
                top_p=0.9
            )[0]

        pred_text = tokenizer.decode(pred_tokens, skip_special_tokens=True)
        length_diff = abs(len(pred_text) - true_length)

        if length_diff < closest_length_diff:
            closest_pred_tokens = pred_tokens
            closest_pred_text = pred_text
            closest_length_diff = length_diff

        if len(pred_text) == true_length:
            return pred_tokens, pred_text

    # 如果超过尝试次数仍未匹配，返回最接近的序列
    print(f"Warning: Maximum attempts reached. Using closest match for input with length difference {closest_length_diff}.")
    return closest_pred_tokens, closest_pred_text

def evaluate(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load tokenizer and model
    tokenizer = T5Tokenizer.from_pretrained("ZYMScott/CodonT5")
    model = T5ForConditionalGeneration.from_pretrained(args.model_name).to(device)

    # Load test data
    test_dataset = load_and_tokenize_data(
        file_path=args.test_data_path,
        tokenizer=tokenizer,
        max_length=args.max_length,
    )

    # Create DataLoader for batching
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model.eval()
    predictions = []
    all_preds = []
    all_labels = []
    with open(args.output_file, 'w') as output_file:
        output_file.write("True RNA Sequence,Predicted RNA Sequence\n")

        with tqdm(test_dataloader) as progress_bar:
            for batch in progress_bar:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to("cpu").numpy()  # 转为 NumPy 以释放显存

                for i in range(len(input_ids)):
                    true_tokens = labels[i]
                    true_text = tokenizer.decode(true_tokens, skip_special_tokens=True) # AUG CCG GUA
                    true_length = len(true_text)

                    
                    pred_tokens, pred_text = generate_consistent_length(
                        model=model,
                        tokenizer=tokenizer,
                        input_ids=input_ids[i].unsqueeze(0),
                        attention_mask=attention_mask[i].unsqueeze(0),
                        true_length=true_length,
                        max_length=args.max_length,
                        device=device,
                        max_attempts=args.max_generate_attempts
                    )
                    all_preds.append(pred_tokens.cpu().numpy())
                    all_labels.append(true_tokens)
                    output_file.write(f"{true_text},{pred_text}\n")

    # 计算 token-level accuracy
    token_accuracy = calculate_token_level_accuracy(all_preds, all_labels)
    print(f"Token-level accuracy: {token_accuracy:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_data_path", type=str, required=True, help="Path to the test data.")
    parser.add_argument("--model_name", type=str, default="t5-base", help="Name or path of the pretrained T5 model.")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length for the model.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for evaluation.")
    parser.add_argument("--output_file", type=str, default="predictions.csv", help="File to save true and predicted sequences.")
    parser.add_argument("--max_generate_attempts", type=int, default=10, help="Maximum attempts for generating consistent-length sequences.")
    args = parser.parse_args()

    evaluate(args)


# CUDA_VISIBLE_DEVICES=1 python evaluate_codont5.py --test_data_path /raid_elmo/home/lr/zym/protein2rna/ncbi_dataset/data/test_flat.json --model_name /raid_elmo/home/lr/zym/protein2rna/checkpoints/codont5/checkpoint-1413000 --batch_size 16