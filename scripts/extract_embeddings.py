import json
import os
import argparse
import torch
import esm

def load_dataset(data_path):
    assert os.path.exists(data_path), f"{data_path} does not exist"

    with open(data_path, "r") as file:
        rows = json.load(file)
    
    return rows


def save_embedding_to_file(embedding_dict, file_path):
    with open(file_path, "a") as file:
        json.dump(embedding_dict, file)
        file.write("\n")  # 换行，以便每个embedding占一行

def extract_and_save_embeddings(args, model, batch_converter, device):
    for split in ['train']:
        category_path = os.path.join(args.data_path, f'{split}_flat.json')
        print(category_path)
        rows = load_dataset(category_path)
        print(f"Loaded {len(rows)} rows")

        embeddings_file_path = os.path.join(args.out_dir, f"{split}_embeddings.jsonl")
        
        for item in rows:
            protein_data = item['prot_sequence']
            batch_labels, batch_strs, batch_tokens = batch_converter([("", protein_data)])
            batch_tokens = batch_tokens.to(device)
            batch_lens = (batch_tokens != model.alphabet.padding_idx).sum(1)

            with torch.no_grad():
                protein_repr = model(batch_tokens, repr_layers=[33], return_contacts=True)
            token_representations = protein_repr["representations"][33]

            sequence_representations = []
            for i, tokens_len in enumerate(batch_lens):
                sequence_representations.append(token_representations[i, 1 : tokens_len - 1].mean(0))

            sequence_embedding = torch.stack(sequence_representations, dim=0).squeeze()
            embedding_dict = {
                "Assembly Accession": item["Assembly Accession"],
                "protein_id": item["protein_id"],
                "embedding": sequence_embedding.cpu().numpy().tolist()  # 转换为列表
            }
            save_embedding_to_file(embedding_dict, embeddings_file_path)

        print(f"Embeddings for {split} saved to {embeddings_file_path}")



def save_progress(progress, progress_file_path):
    """保存当前进度到文件"""
    with open(progress_file_path, "w") as file:
        file.write(str(progress))

def load_progress(progress_file_path):
    """从文件加载当前进度"""
    try:
        with open(progress_file_path, "r") as file:
            return int(file.read())
    except FileNotFoundError:
        return 0  # 如果文件不存在，从头开始

def extract_and_save_embeddings_with_progress(args, model, batch_converter, device):
    for split in ['train']:
        category_path = os.path.join(args.data_path, f'{split}_flat.json')
        rows = load_dataset(category_path)
        print(f"Loaded {len(rows)} rows")

        embeddings_file_path = os.path.join(args.out_dir, f"{split}_embeddings.jsonl")
        progress_file_path = os.path.join(args.out_dir, f"{split}_progress.txt")
        current_progress = load_progress(progress_file_path)
        
        for index, item in enumerate(rows[current_progress:], start=current_progress):
            protein_data = item['prot_sequence']
            batch_labels, batch_strs, batch_tokens = batch_converter([("", protein_data)])
            batch_tokens = batch_tokens.to(device)
            batch_lens = (batch_tokens != model.alphabet.padding_idx).sum(1)

            with torch.no_grad():
                protein_repr = model(batch_tokens, repr_layers=[33], return_contacts=True)
            token_representations = protein_repr["representations"][33]

            sequence_representations = []
            for i, tokens_len in enumerate(batch_lens):
                sequence_representations.append(token_representations[i, 1 : tokens_len - 1].mean(0))

            sequence_embedding = torch.stack(sequence_representations, dim=0).squeeze()
            embedding_dict = {
                "Assembly Accession": item["Assembly Accession"],
                "protein_id": item["protein_id"],
                "embedding": sequence_embedding.cpu().numpy().tolist()  # 转换为列表
            }
            save_embedding_to_file(embedding_dict, embeddings_file_path)
            save_progress(index + 1, progress_file_path)

        print(f"Embeddings for {split} saved to {embeddings_file_path}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--num_proc", type=int, default=1)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, alphabet = esm.pretrained.load_model_and_alphabet("/raid_elmo/home/lr/zym/protein2rna/checkpoints/esm2_t33_650M_UR50D.pt")
    batch_converter = alphabet.get_batch_converter()

    model.to(device)
    model.eval()

    extract_and_save_embeddings_with_progress(args, model, batch_converter, device)
