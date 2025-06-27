from typing import List, Dict, Sequence
from dataclasses import dataclass, field
import logging
import json
import os
import re
import numpy as np
import pandas as pd
from datasets import load_from_disk, load_dataset, Dataset as HFDataset
import transformers
import torch
import tqdm
from torch.utils.data import Dataset, DataLoader
# from pro2rna.training import ModelArguments
from pro2rna.training_data import _resolve_dataset, GCFDataset, DataArguments, custom_collate_fn, convert_to_codon_seq, mytok
from pro2rna.training import ModelArguments
from pro2rna.model.model import RevProtein
from pro2rna.utils.tokenizer import get_base_tokenizer


amino_acids_codons = {
    'A': ['GCT', 'GCC', 'GCA', 'GCG'], # Alanine
    'R': ['CGT', 'CGC', 'CGA', 'CGG', 'AGA', 'AGG'], # Arginine
    'N': ['AAT', 'AAC'], # Asparagine
    'D': ['GAT', 'GAC'], # Aspartic acid
    'C': ['TGT', 'TGC'], # Cysteine
    'E': ['GAA', 'GAG'], # Glutamic acid
    'Q': ['CAA', 'CAG'], # Glutamine
    'G': ['GGT', 'GGC', 'GGA', 'GGG'], # Glycine
    'H': ['CAT', 'CAC'], # Histidine
    'I': ['ATT', 'ATC', 'ATA'], # Isoleucine
    'L': ['TTA', 'TTG', 'CTT', 'CTC', 'CTA', 'CTG'], # Leucine
    'K': ['AAA', 'AAG'], # Lysine
    'M': ['ATG'], # Methionine
    'F': ['TTT', 'TTC'], # Phenylalanine
    'P': ['CCT', 'CCC', 'CCA', 'CCG'], # Proline
    'S': ['TCT', 'TCC', 'TCA', 'TCG', 'AGT', 'AGC'], # Serine
    'T': ['ACT', 'ACC', 'ACA', 'ACG'], # Threonine
    'W': ['TGG'], # Tryptophan
    'Y': ['TAT', 'TAC'], # Tyrosine
    'V': ['GTT', 'GTC', 'GTA', 'GTG'], # Valine
    '*': ['TAA', 'TAG', 'TGA'] # Stop codon
}

@dataclass
class EvaluationArguments(ModelArguments):
    output_dir: str = field(default=None, metadata={"help": "Path to the output file."})
    verbose: bool = field(default=False, metadata={"help": "Print verbose output."})
    esm_name_or_path: str = field(
        default="esm2_t33_650M_UR50D", metadata={"help": "Name or Path to the esm model"}
    )
    species_model: str = field(default="scibert")
    model_path: str = field(
        default=None, metadata={"help": "Path to rest model"}
    )
    use_codon_feature: bool = field(
        default=False,
        metadata={
            "help": "Whether use extra codon feature to train"
        },
    )
    refinement: bool = field(
        default=False,
        metadata={
            "help": "Whether use codon optimization"
        },
    )
    num_refinement_layers: int = field(default=2)
    num_optimization_layers: int = field(default=6)
    protein_file_path: str = field(
        default=None, metadata={"help": "Path to the protein data."}
    )
    cds_file_path: str = field(
        default=None, metadata={"help": "Path to the cds data."}
    )
    species_file_path: str = field(
        default=None, metadata={"help": "Path to the species information data."}
    )



class InferenceGCFDataset(Dataset):
    def __init__(
        self,
        dataset,
        max_length:int = 512,
    ):
        super(InferenceGCFDataset, self).__init__()
        self.dataset = dataset
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i) -> Dict:
        item = self.dataset[i]
        protein_sequence = item["prot_sequence"]
        if len(protein_sequence) > self.max_length-2:
            protein_sequence = protein_sequence[:self.max_length-2]  # reserve for <cls> and <eos>
        else:
            protein_sequence = protein_sequence.ljust(self.max_length-2, '-')
        cDNA_sequence = item['cds_sequence'][: 3 * (self.max_length-1)]
        
        codon_seqs, codon_ids = convert_to_codon_seq(cDNA_sequence, max_length=self.max_length)
        codon_ids = torch.tensor([codon_ids], dtype=torch.int64) 

        return {
            "Assembly Accession": item["Assembly Accession"],
            "protein_sequence": protein_sequence,   # max_length - 2
            "labels": codon_ids,                    # max_length
            "cds_sequence": codon_seqs,
            "Order": item['Order'],
            "Family": item['Family'],
            "Genus": item['Genus'],
            "Species": item['Species'],
            "scientific_name": item['Organism Scientific Name'],
        }

def load_trained_model(config):
    model = RevProtein(config)
    model.load_model_weights(config.model_path, strict=False)
    model.eval()
    return model

def build_prompt(row):
    return f"The organism {row['Organism Scientific Name']} belongs to the order {row['Order']}, family {row['Family']}, genus {row['Genus']}, and species {row['Species']}."


def check_same_protein(real_protein_seq, predicted_protein_seq):
    if predicted_protein_seq.endswith('*'):
        predicted_protein_seq = predicted_protein_seq[:-1]
    if real_protein_seq != predicted_protein_seq:
        return False
    return True

def extract_protein_id(s):
    # 定义正则表达式匹配模式
    pattern = r'protein_id=([^ \]]+)'
    # 搜索字符串
    match = re.search(pattern, s)
    # 如果找到匹配项，则返回匹配的值
    if match:
        return match.group(1)  # group(1) 返回匹配的值，group(0) 会返回整个匹配的字符串
    else:
        return None


def read_protein_sequences(file_path):
    protein_sequences = {}
    with open(file_path, 'r') as file:
        protein_id = ''
        sequence = ''
        for line in file:
            if line.startswith('>'):
                if protein_id:
                    protein_sequences[protein_id] = sequence
                protein_id = line.split()[0][1:]
                sequence = ''
            else:
                sequence += line.strip()
        # Don't forget the last entry
        if protein_id:
            protein_sequences[protein_id] = sequence
    sorted_protein_sequences = {protein_id: protein_sequences[protein_id] for protein_id in sorted(protein_sequences)}
    
    return sorted_protein_sequences


def read_codon_sequences(file_path):
    codon_sequences = {}
    with open(file_path, 'r') as file:
        protein_id = ''
        sequence = ''
        for line in file:
            if line.startswith('>'):
                if protein_id:
                    codon_sequences[protein_id] = sequence
                protein_id = extract_protein_id(line)
                sequence = ''
            else:
                sequence += line.strip()
        # Don't forget the last entry
        if protein_id:
            codon_sequences[protein_id] = sequence
    
    sorted_codon_sequences = {protein_id: codon_sequences[protein_id] for protein_id in sorted(codon_sequences)}
    
    return sorted_codon_sequences

def check_standard_protein_seq(protein_sequence):
    for aa in protein_sequence:
        if aa not in amino_acids_codons:
            return False
    return True

def check_length(cds_seq_len, prot_seq_len):
    return cds_seq_len % 3 == 0 and (prot_seq_len + 1) * 3 == cds_seq_len


def check_valid_cds(cds_seq):
    for codon in cds_seq:
        if codon not in "ATCG":
            print(f"Invalid DNA seq!! {cds_seq}")
            return False
    return True

def get_dataset(protein_file_path, cds_file_path, species_file_path):    
    species_information = pd.read_csv(species_file_path)
    results = []
    protein_seq = read_protein_sequences(protein_file_path)
    cds_seq = read_codon_sequences(cds_file_path)
    
    for protein_id, prot_seq in protein_seq.items():
        
        cds_seq_for_id = cds_seq.get(protein_id, None)
        if cds_seq_for_id and cds_seq_for_id.startswith("ATG") and check_valid_cds(cds_seq_for_id) and check_standard_protein_seq(prot_seq) and check_length(len(cds_seq_for_id), len(prot_seq)):
            record = {}
            record['Assembly Accession'] = species_information['Accession'].iloc[0]
            record['Organism Scientific Name'] = species_information['Organism Name'].iloc[0]
            record['Kindom'] = species_information['kingdom'].iloc[0]
            record['Phylum'] = species_information['phylum'].iloc[0]
            record['Class'] = species_information['class'].iloc[0]
            record['Order'] = species_information['order'].iloc[0]
            record['Family'] = species_information['family'].iloc[0]
            record['Genus'] = species_information['genus'].iloc[0] 
            record['Species'] = species_information['species'].iloc[0] 


            record['prot_sequence'] = prot_seq
            record['cds_sequence'] = cds_seq_for_id
            record['protein_id'] = protein_id  
            
            results.append(record)
    
    return results

def _evaluate(model, test_loader, eval_args, device):
    gcf_accuracies = {}
    total_correct_predictions = 0
    total_predictions = 0

    codon_alphabet = model.codon_alphabet
    # tokenizer = get_base_tokenizer()
    # vocab = tokenizer.get_vocab()

    inv_codon_alphabet= {v: k for k, v in codon_alphabet.tok_to_idx.items()}
    pad_token_id = codon_alphabet.padding_idx
    test_loader = tqdm.tqdm(test_loader, desc="Evaluating", total=len(test_loader))

    output_dir = eval_args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    real_sequences_file_path = os.path.join(output_dir, 'real_sequences.txt')
    predicted_sequences_file_path = os.path.join(output_dir, 'predicted_sequences.txt')

    real_sequences_file = open(real_sequences_file_path, 'a')
    predicted_sequences_file = open(predicted_sequences_file_path, 'a')

    # if eval_args.species_model == "":
    #     pass
    for entry in test_loader:
        gcf_id = entry['Assembly Accession'][0]
        # print(gcf_id, " ", type(gcf_id))
        cds_sequence = entry['cds_sequence'][0]
        protein_sequence = entry['protein_sequence'][0]
        prompt = entry['prompts'][0]
        labels = entry.pop('labels')
        # labels = entry['labels']
        with torch.no_grad():
            outputs = model(labels=labels, **entry)
        logits = outputs.get('logits')
        predicted_classes = torch.argmax(logits, dim=-1)
        
        # correct_predictions = (predicted_classes == labels).float() 

        # 忽略PAD标记的位置
        mask = labels != pad_token_id
        # correct_predictions = (predicted_classes[mask].to('cpu') == labels[mask].to('cpu')).float()
        correct_predictions = (predicted_classes[mask].detach().cpu().numpy() == labels[mask].detach().cpu().numpy())

        # 累计GCF种类准确率
        if gcf_id not in gcf_accuracies:
            gcf_accuracies[gcf_id] = {'correct': 0, 'total': 0}
        gcf_accuracies[gcf_id]['correct'] += correct_predictions.sum().item()
        gcf_accuracies[gcf_id]['total'] += mask.sum().item()

        # 累计总体准确率
        total_correct_predictions += correct_predictions.sum().item()
        total_predictions += mask.sum().item()

        # 计算准确率
        accuracy = total_correct_predictions / total_predictions
        # print(f"accuracy: {accuracy}")
        real_cds_sequences = [seq for seq in cds_sequence if seq != '<pad>']
        real_sequence_str = ' '.join(real_cds_sequences)

        predicted_sequence = [inv_codon_alphabet[idx.item()] for idx in predicted_classes[mask]]
        predicted_sequence_str = ' '.join(predicted_sequence[:len(real_cds_sequences)])
        
        real_sequences_file.write(f"{gcf_id}\t{real_sequence_str}\n")
        predicted_sequences_file.write(f"{gcf_id}\t{predicted_sequence_str}\n")

        # 计算每个GCF种类的准确率并保存
    for gcf_id in gcf_accuracies:
        gcf_accuracies[gcf_id]['accuracy'] = gcf_accuracies[gcf_id]['correct'] / gcf_accuracies[gcf_id]['total']

    overall_accuracy = total_correct_predictions / total_predictions
    gcf_accuracies['overall'] = overall_accuracy

    print(f"Accuracy: {overall_accuracy}")
    real_sequences_file.close()
    predicted_sequences_file.close()

    return gcf_accuracies
   

def _save_score(score: dict, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "score_mlp_2_10000.json"), "w") as file:
        json.dump(score, file, indent=4)


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser = transformers.HfArgumentParser((EvaluationArguments, DataArguments))
    eval_args, data_args, _ = parser.parse_args_into_dataclasses(return_remaining_strings=True)



    dataset = get_dataset(eval_args.protein_file_path, eval_args.cds_file_path, eval_args.species_file_path)


    print(eval_args, data_args)
    

    test_dataset = InferenceGCFDataset(dataset)

    # print(test_dataset[0])
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=custom_collate_fn)
    model = load_trained_model(eval_args).to(device)
    score = _evaluate(model, test_loader, eval_args, device)
   
    if eval_args.output_dir:
        _save_score(score, eval_args.output_dir)


