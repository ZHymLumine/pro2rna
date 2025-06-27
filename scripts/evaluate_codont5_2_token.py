import argparse
import torch
from torch.utils.data import DataLoader
from datasets import Dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration
import json
from tqdm import tqdm


# def load_and_tokenize_data(file_path, input_tokenizer, label_tokenizer, max_length=512):
#     input_texts = []
#     labels = []

#     with open(file_path, 'r') as f:
#         data_list = json.load(f)

#     for data in data_list:
#         taxonomy = (
#             f"<{data['Kindom']}><{data['Phylum']}><{data['Class']}>"
#             f"<{data['Order']}><{data['Family']}><{data['Genus']}><{data['Species']}>"
#         )

#         prot_sequence = data["prot_sequence"]
#         input_text = f"{taxonomy}{prot_sequence}"
#         input_texts.append({"text": input_text})

#         rna_sequence = data["cds_sequence"].replace("T", "U")
#         labels.append({"text": rna_sequence})

#     # 对输入进行 tokenization
#     def tokenize_inputs(examples):
#         model_inputs = input_tokenizer(
#             examples["text"],
#             truncation=True,
#             padding="max_length",
#             max_length=max_length
#         )
#         return model_inputs

#     # 对标签进行 tokenization
#     def tokenize_labels(examples):
#         model_labels = label_tokenizer(
#             examples["text"],
#             truncation=True,
#             padding="max_length",
#             max_length=max_length
#         )
#         return model_labels

#     input_dataset = Dataset.from_list(input_texts)
#     label_dataset = Dataset.from_list(labels)

#     input_tokenized = input_dataset.map(tokenize_inputs, batched=True, num_proc=6, remove_columns=["text"])
#     input_tokenized.set_format("torch", columns=["input_ids", "attention_mask"])

#     label_tokenized = label_dataset.map(tokenize_labels, batched=True, num_proc=6, remove_columns=["text"])
#     label_tokenized.set_format("torch", columns=["input_ids"])

#     true_labels_tensor = torch.stack([torch.tensor(x["input_ids"]).clone().detach() for x in label_tokenized])
#     return input_tokenized, true_labels_tensor, label_tokenized

def load_and_tokenize_data(file_path, input_tokenizer, label_tokenizer, max_length=512):
    input_texts = []
    labels = []
    
    with open(file_path, 'r') as f:
        data_list = json.load(f)

    for data in data_list:
        taxonomy = (
            f"<{data['Kindom']}><{data['Phylum']}><{data['Class']}>"
            f"<{data['Order']}><{data['Family']}><{data['Genus']}><{data['Species']}>"
        )
        
        # 构建输入文本（taxonomy + prot_sequence）
        prot_sequence = data["prot_sequence"]
        input_text = f"{taxonomy}{prot_sequence}"
        input_texts.append({"text": input_text})
        
        rna_sequence = data["cds_sequence"].replace("T", "U")
        labels.append({"text": rna_sequence})

    def tokenize_inputs(examples):
        return input_tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length
        )

    def tokenize_labels(examples):
        labels = label_tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length
        )["input_ids"]

        # Mask padding tokens in the labels
        # labels = [[-100 if token_id == label_tokenizer.pad_token_id else token_id for token_id in label] for label in labels]
        return {"labels": labels}

    input_dataset = Dataset.from_list(input_texts)
    label_dataset = Dataset.from_list(labels)

    input_tokenized = input_dataset.map(tokenize_inputs, batched=True, num_proc=4, remove_columns=["text"])
    input_tokenized.set_format("torch", columns=["input_ids", "attention_mask"])

    label_tokenized = label_dataset.map(tokenize_labels, batched=True, num_proc=4, remove_columns=["text"])
    labels_column = label_tokenized["labels"]

    input_tokenized = input_tokenized.add_column("labels", labels_column)
    input_tokenized.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    return input_tokenized

def decode_logits_and_labels(logits, label_tokenizer, label_ids):
    predicted_ids = torch.argmax(logits, dim=-1)
    truncated_predicted_ids = []
    truncated_label_ids = []

    # print(f"predicted_ids :{predicted_ids}")
    # print(f"predicted_ids shape :{predicted_ids.shape}")

    # print(f"label_ids :{label_ids.shape}")
    # print(f"label_ids shape :{label_ids.shape}")
    for i, sequence in enumerate(predicted_ids):
        label_seq = label_ids[i].tolist()

        # 找到第一个 </s> (token ID=1) 的位置
        if 1 in label_seq:
            valid_length = label_seq.index(1)
        else:
            valid_length = len(label_seq)

        # print(valid_length)
        truncated_predicted_sequence = sequence[:valid_length].tolist()
        truncated_label_sequence = label_seq[:valid_length]

        truncated_predicted_ids.append(truncated_predicted_sequence)
        truncated_label_ids.append(truncated_label_sequence)

    special_token_ids = list(label_tokenizer.all_special_ids) 
    if any(token in special_token_ids for token in truncated_predicted_sequence):
        print(f"Special token found in predicted sequence: {truncated_predicted_sequence}")
        print(f"label sequence:: {truncated_label_sequence}")
        if len(truncated_predicted_sequence) != len(truncated_label_sequence):
            print("truncated id length different!!!")
    if any(token in special_token_ids for token in truncated_label_sequence):
        print(f"Special token found in label sequence: {truncated_label_sequence}")

    decoded_predicted_text = label_tokenizer.batch_decode(
        truncated_predicted_ids, skip_special_tokens=False
    )
    decoded_label_text = label_tokenizer.batch_decode(
        truncated_label_ids, skip_special_tokens=False
    )
    cleaned_predicted_text = [text.replace(" ", "").replace("\n", "") for text in decoded_predicted_text]
    cleaned_label_text = [text.replace(" ", "").replace("\n", "") for text in decoded_label_text]

    return predicted_ids, cleaned_predicted_text, cleaned_label_text

import numpy as np

def calculate_token_level_accuracy_fast(all_preds, all_labels, pad_token_id):
    """
    高效计算 token-level accuracy，排除 pad token 的影响。

    :param all_preds: 所有预测结果（list of numpy arrays）。
    :param all_labels: 所有真实标签（list of numpy arrays）。
    :param pad_token_id: 填充值的 token ID。
    :return: Token-level accuracy。
    """
    preds_concat = np.concatenate(all_preds, axis=0)
    labels_concat = np.concatenate(all_labels, axis=0)

    non_pad_mask = labels_concat != pad_token_id

    total_tokens = np.sum(non_pad_mask)
    matched_tokens = np.sum((preds_concat == labels_concat) & non_pad_mask)

    token_accuracy = matched_tokens / total_tokens if total_tokens > 0 else 0
    return token_accuracy


def evaluate(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load input and label tokenizers
    input_tokenizer = T5Tokenizer.from_pretrained("ZYMScott/CodonT5_input")
    label_tokenizer = T5Tokenizer.from_pretrained("ZYMScott/CodonT5_label")

    model = T5ForConditionalGeneration.from_pretrained(args.model_name).to(device)

    # Load model
    model.resize_token_embeddings(len(input_tokenizer))
    model.lm_head = torch.nn.Linear(
        model.config.d_model,
        len(label_tokenizer),
        bias=False
    ).to(device) 

    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict, strict=True)

    print(f"model:{model}")

    test_dataset = load_and_tokenize_data(
        file_path=args.test_data_path,
        input_tokenizer=input_tokenizer,
        label_tokenizer=label_tokenizer,
        max_length=args.max_length,
    )

    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model.eval()

    with open(args.output_file, 'w') as output_file:
        output_file.write("True RNA Sequence,Predicted RNA Sequence\n")

        all_preds = []
        all_labels = []
        for batch_idx, batch in enumerate(tqdm(test_dataloader)):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            # start_idx = batch_idx * args.batch_size
            # end_idx = start_idx + input_ids.size(0)
            # true_label_batch = true_labels[start_idx:end_idx].to(device)
            # print(f'attention_mask: {attention_mask}')
            # print(f'attention_mask : {attention_mask.shape}')
            
            # print(f'labels: {labels}')
            # print(f'labels: {labels.shape}')
            
            decoder_input_ids = torch.cat([
                torch.full((input_ids.size(0), 1), label_tokenizer.pad_token_id, dtype=torch.long, device=device),
                labels[:, :-1].to(device)
            ], dim=1)

            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids)
                logits = outputs.logits

            # Decode logits and calculate accuracy
            pred_ids, pred_texts, true_texts_decoded = decode_logits_and_labels(
                logits, label_tokenizer, labels
            )

            pred_ids = pred_ids.detach().cpu().numpy()
            labels = labels.detach().cpu().numpy()

            all_preds.append(pred_ids)
            all_labels.append(labels)

            for pred_text, true_text in zip(pred_texts, true_texts_decoded):
                # truncated_true_text = true_text[:len(pred_text)]
                # if len(pred_text) != len(true_text):
                    # print(f'different length at idx: {batch_idx}')
                    # print(f"true: {len(true_text)}, pred: {len(pred_text)}")
                output_file.write(f"{true_text},{pred_text}\n")

        token_accuracy = calculate_token_level_accuracy_fast(all_preds, all_labels, label_tokenizer.pad_token_id)
        print(f"Token-level accuracy: {token_accuracy:.4f}")
        print("Evaluation completed and saved to", args.output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_data_path", type=str, required=True, help="Path to the test data.")
    parser.add_argument("--model_name", type=str, default="t5-base", help="Name or path of the pretrained T5 model.")
    parser.add_argument("--checkpoint", type=str, default="e", help="path of the pretrained T5 model.")

    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length for the model.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for evaluation.")
    parser.add_argument("--output_file", type=str, default="predictions_length_label.csv", help="File to save true and predicted sequences.")
    args = parser.parse_args()

    evaluate(args)

# CUDA_VISIBLE_DEVICES=2 python evaluate_codont5_2_token.py --test_data_path /raid_elmo/home/lr/zym/protein2rna/ncbi_dataset/data/test_flat.json --model_name t5-small --checkpoint /raid_elmo/home/lr/zym/protein2rna/checkpoints/codont5_2_tokenizer/checkpoint-22000/pytorch_model.bin --batch_size 16