from dataclasses import dataclass, field
import logging
import json
import os
import numpy as np
from datasets import load_from_disk, load_dataset, Dataset as HFDataset
import transformers
import torch
import tqdm
from torch.utils.data import DataLoader
# from pro2rna.training import ModelArguments
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR, set_seed
from pro2rna.training_data import _resolve_dataset, GCFDataset, DataArguments, custom_collate_fn
from pro2rna.training import ModelArguments
from pro2rna.model.model import RevProtein
from pro2rna.utils.tokenizer import get_base_tokenizer


codon_table = {
    'UUU': 'F', 'UUC': 'F', 'UUA': 'L', 'UUG': 'L', 
    'CUU': 'L', 'CUC': 'L', 'CUA': 'L', 'CUG': 'L', 
    'AUU': 'I', 'AUC': 'I', 'AUA': 'I', 'AUG': 'M', 
    'GUU': 'V', 'GUC': 'V', 'GUA': 'V', 'GUG': 'V', 
    'UCU': 'S', 'UCC': 'S', 'UCA': 'S', 'UCG': 'S', 
    'CCU': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P', 
    'ACU': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T', 
    'GCU': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A', 
    'UAU': 'Y', 'UAC': 'Y', 'UAA': '*', 'UAG': '*', 
    'CAU': 'H', 'CAC': 'H', 'CAA': 'Q', 'CAG': 'Q', 
    'AAU': 'N', 'AAC': 'N', 'AAA': 'K', 'AAG': 'K', 
    'GAU': 'D', 'GAC': 'D', 'GAA': 'E', 'GAG': 'E', 
    'UGU': 'C', 'UGC': 'C', 'UGA': '*', 'UGG': 'W', 
    'CGU': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R', 
    'AGU': 'S', 'AGC': 'S', 'AGA': 'R', 'AGG': 'R', 
    'GGU': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G'
}

@dataclass
class EvaluationArguments(ModelArguments):
    # dataset_path: str = field(
    #     default=None, metadata={"help": "Path to the training data."}
    # )
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


def load_trained_model(
   config
):
    model = RevProtein(config)
    model.load_model_weights(config.model_path, strict=False)
    model.eval()
    return model


def check_same_protein(real_protein_seq, predicted_protein_seq):
    if predicted_protein_seq.endswith('*'):
        predicted_protein_seq = predicted_protein_seq[:-1]
    if real_protein_seq != predicted_protein_seq:
        return False
    return True


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
        labels = entry.pop('labels')
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
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser = transformers.HfArgumentParser((EvaluationArguments, DataArguments))
    eval_args, data_args, _ = parser.parse_args_into_dataclasses(return_remaining_strings=True)


    print("eval args: ", eval_args)
    print("data args:", data_args)
    
    model = load_trained_model(
        eval_args
    ).to(device)

    test_dataset = GCFDataset(data_args, split='test')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=custom_collate_fn)
    score = _evaluate(model, test_loader, eval_args, device)

   
    if eval_args.output_dir:
        _save_score(score, eval_args.output_dir)


