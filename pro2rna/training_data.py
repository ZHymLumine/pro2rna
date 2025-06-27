from typing import List, Dict, Sequence
from dataclasses import dataclass, field
import logging
import os
import json
import pandas as pd

from torch.utils.data import Dataset, DataLoader
from datasets import load_from_disk, load_dataset, Dataset as HFDataset
import transformers
from transformers import AutoTokenizer, AutoModel
import torch
from pro2rna.model.model import RevProtein
from pro2rna.calm.alphabet import Alphabet

@dataclass
class DataArguments:
    dataset_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )


def _resolve_dataset(path: str, split:str) -> HFDataset:
    path = os.path.join(path, split)
    if os.path.exists(path):
        return load_from_disk(path)
    else:
        return load_dataset(path, split="train", data_files="*.arrow")
    
def mytok(seq, kmer_len, s):
    """
    Tokenize a sequence into kmers.

    Args:
        seq (str): sequence to tokenize
        kmer_len (int): length of kmers
        s (int): increment
    """
    seq = seq.upper().replace("T", "U")
    kmer_list = []
    for j in range(0, (len(seq) - kmer_len) + 1, s):
        kmer_list.append(seq[j : j + kmer_len])
    return kmer_list


def convert_to_codon_seq(cds_seq, max_length=512):
    """
    Convert a DNA sequence to a list of RNA codons and padding.

    Args:
        cds_seq (str): DNA sequence to be converted.
        max_length (int): Maximum length of the codon sequence list.

    Returns:
        list: A list of RNA codons padded to max_length.
    """
    alphabet = Alphabet.from_architecture("CodonModel")
    cds_seq = cds_seq.upper().replace("T", "U")
    lst_tok = mytok(cds_seq, 3, 3)
    
    if len(lst_tok) > max_length - 2:
        lst_tok = lst_tok[:max_length - 2] + ['<eos>']
    lst_tok = ['<cls>'] + lst_tok
    lst_tok += ['<pad>'] * (max_length - len(lst_tok))
    
    codon_ids = alphabet.encode(' '.join(lst_tok))
    return lst_tok, codon_ids


def tokenize_codon(cds_seq, tokenizer, max_length=512, special_tokens=None):
    """
    Convert a DNA sequence to a list of RNA codons and padding.

    Args:
        cds_seq (str): DNA sequence to be converted.
        max_length (int): Maximum length of the codon sequence list.

    Returns:
        list: A list of RNA codons padded to max_length.
    """
    if special_tokens is None:
        special_tokens = {'cls_token': '@', 'pad_token': '#', 'eos_token': '$'}
    cds_seq = cds_seq.upper().replace("T", "U")
    lst_tok = mytok(cds_seq, 3, 3)
    
    if len(lst_tok) > max_length - 2:
        lst_tok = lst_tok[:max_length - 2] + [special_tokens['eos_token']]
    lst_tok = [special_tokens['cls_token']] + lst_tok
    lst_tok += [special_tokens['pad_token']] * (max_length - len(lst_tok))
    
    codon_ids = tokenizer.encode(' '.join(lst_tok))
    return lst_tok, codon_ids

def build_prompt(row):
    return f"The organism {row['Organism Scientific Name']} belongs to the order {row['Order']}, family {row['Family']}, genus {row['Genus']}, and species {row['Species']}."

def custom_collate_fn(batch):
    assembly_accessions = []
    protein_sequences = []
    cDNA_sequences = []
    protein_ids = []
    codon_labels = []
    codon_seqs = []
    prompts = []

    for i, item in enumerate(batch):
        assembly_accessions.append(item["Assembly Accession"])
        row = {
            'Organism Scientific Name': item['scientific_name'],
            'Order': item['Order'],
            'Family': item['Family'],
            'Genus': item['Genus'],
            'Species': item['Species'] 
        }
        prompts.append(build_prompt(row))

        # collate protein and dna information
        protein_sequences.append(item["protein_sequence"])
        # protein_ids.append(item["protein_id"])
        # cDNA_sequences.append(item['cDNA_sequence'])

        codon_seqs.append(item['cds_sequence'])
        codon_labels.append(item["labels"].squeeze())
        

        codon_labels_tensors = [item.clone().detach() for item in codon_labels]
        codon_labels_tensors = torch.stack(codon_labels_tensors, dim=0)

        #  for next token prediction
        # target_seqs = codon_labels_tensors[:, :-1]
        # target_labels = codon_labels_tensors[:, 1:]

    
    return {
        "Assembly Accession": assembly_accessions,
        "protein_sequence": protein_sequences,
        "labels": codon_labels_tensors,
        "prompts": prompts,
        "cds_sequence": codon_seqs,
    }


class GCFDataset(Dataset):
    def __init__(
        self,
        data_args: DataArguments,  
        split:str,   
        max_length:int = 512,
        tokenizer: AutoTokenizer = None,
    ):
        super(GCFDataset, self).__init__()
        self.dataset = _resolve_dataset(data_args.dataset_path, split)

        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i) -> Dict:
        item = self.dataset[i]
        protein_sequence = item["protein_sequence"]
        if len(protein_sequence) > self.max_length-2:
            protein_sequence = protein_sequence[:self.max_length-2]  # reserve for <cls> and <eos>
        else:
            protein_sequence = protein_sequence.ljust(self.max_length-2, '-')
        cDNA_sequence = item['cds_sequence'][: 3 * (self.max_length-1)]
        
        codon_seqs, codon_ids = convert_to_codon_seq(cDNA_sequence, max_length=self.max_length)
        codon_ids = torch.tensor(codon_ids, dtype=torch.int64)

        return {
            "Assembly Accession": item["gcf_id"],
            "protein_sequence": protein_sequence,   # max_length - 2
            "labels": codon_ids,                    # max_length
            "cds_sequence": codon_seqs,
            "Order": item['order'],
            "Family": item['family'],
            "Genus": item['genus'],
            "Species": item['species'],
            "scientific_name": item['organism_name'],
        }