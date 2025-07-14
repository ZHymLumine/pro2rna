#!/usr/bin/env python3
"""
Inference script for generating mRNA sequences from protein sequences using fine-tuned model.
Given GFP protein sequence and taxonomy information (e.g., human), generate mRNA sequence optimized for that organism.
"""

import argparse
import torch
import pandas as pd
import json
import os
from typing import List, Dict
import numpy as np
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from pro2rna.model.model import RevProtein
from pro2rna.calm.alphabet import Alphabet
from pro2rna.training_data import build_prompt


def parse_args():
    parser = argparse.ArgumentParser(description="Generate mRNA sequences using fine-tuned Pro2RNA model")
    
    # Model arguments
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the fine-tuned model weights (LoRA)")
    parser.add_argument("--esm_name_or_path", type=str, default="esm2_t33_650M_UR50D",
                        help="ESM model name or path")
    parser.add_argument("--species_model", type=str, default="scibert",
                        help="Species model name")
    parser.add_argument("--decoder_type", type=str, default="RNAdecoder",
                        help="Decoder type")
    parser.add_argument("--RNA_config_path", type=str, required=True,
                        help="Path to RNA decoder config file")
    parser.add_argument("--decoder_path", type=str, required=True,
                        help="Path to RNA decoder checkpoint")
    
    # Input arguments
    parser.add_argument("--protein_fasta", type=str, required=True,
                        help="Path to protein sequences FASTA file")
    parser.add_argument("--species_csv", type=str, required=True,
                        help="Path to CSV file containing species information")
    
    # Generation arguments
    parser.add_argument("--num_samples", type=int, default=10,
                        help="Number of samples to generate per protein")
    parser.add_argument("--max_length", type=int, default=512,
                        help="Maximum sequence length")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=50,
                        help="Top-k sampling")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="Top-p (nucleus) sampling")
    
    # Output arguments
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for generated sequences")
    
    return parser.parse_args()


class ModelArguments:
    def __init__(self, esm_name_or_path, species_model, decoder_type, RNA_config_path, decoder_path):
        self.esm_name_or_path = esm_name_or_path
        self.species_model = species_model
        self.decoder_type = decoder_type
        self.RNA_config_path = RNA_config_path
        self.decoder_path = decoder_path
        
        # Model configuration defaults
        self.embedding_size = 768  # for esm2_t33_650M_UR50D
        self.hidden_dim = 3072
        self.num_heads = 12
        self.num_decoder_layers = 12
        self.max_length = 512
        self.latent_embed_dim = 256
        self.temp = 0.07


def load_model(args):
    """Load the fine-tuned model"""
    print("Loading model...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create model arguments
    model_args = ModelArguments(
        esm_name_or_path=args.esm_name_or_path,
        species_model=args.species_model,
        decoder_type=args.decoder_type,
        RNA_config_path=args.RNA_config_path,
        decoder_path=args.decoder_path
    )
    
    # Initialize model
    model = RevProtein(model_args)
    
    # Move model to device
    model = model.to(device)
    
    # Load fine-tuned weights
    print(f"Loading fine-tuned weights from {args.model_path}")
    model.load_model_weights(args.model_path, strict=False)
    
    # Set to evaluation mode
    model.eval()
    
    return model


def load_protein_sequences(fasta_path: str) -> Dict[str, str]:
    """Load protein sequences from FASTA file"""
    proteins = {}
    for record in SeqIO.parse(fasta_path, "fasta"):
        proteins[record.id] = str(record.seq)
    return proteins


def load_species_info(csv_path: str) -> pd.DataFrame:
    """Load species information from CSV file"""
    return pd.read_csv(csv_path)


def prepare_species_prompt(species_row: pd.Series) -> str:
    """Prepare species prompt from taxonomy information"""
    return build_prompt({
        'Organism Scientific Name': species_row.get('organism_name', species_row.get('scientific_name', '')),
        'Order': species_row.get('order', ''),
        'Family': species_row.get('family', ''),
        'Genus': species_row.get('genus', ''),
        'Species': species_row.get('species', '')
    })


def ids_to_codons(codon_ids: torch.Tensor, alphabet: Alphabet) -> List[str]:
    """Convert codon IDs back to codon sequences"""
    # Convert tensor to list if necessary
    if isinstance(codon_ids, torch.Tensor):
        id_list = codon_ids.cpu().numpy().tolist()
    else:
        id_list = codon_ids
    
    # Convert IDs to tokens using get_tok method
    codons = [alphabet.get_tok(id) for id in id_list]
    
    # Remove special tokens and padding
    filtered_codons = []
    for codon in codons:
        if codon not in ['<cls>', '<pad>', '<eos>', '<unk>', '<mask>']:
            filtered_codons.append(codon)
        elif codon == '<eos>':
            break
    
    return filtered_codons


def codons_to_mrna(codons: List[str]) -> str:
    """Convert list of codons to mRNA sequence"""
    return ''.join(codons)


@torch.no_grad()
def generate_mrna_sequences(model, protein_seq: str, species_prompt: str, 
                           num_samples: int, max_length: int, 
                           temperature: float = 1.0, top_k: int = 50, top_p: float = 0.9) -> List[str]:
    """Generate mRNA sequences for given protein and species"""
    
    # Prepare input
    device = next(model.parameters()).device
    alphabet = Alphabet.from_architecture("CodonModel")
    
    # Truncate protein sequence if too long
    if len(protein_seq) > max_length - 2:
        protein_seq = protein_seq[:max_length - 2]
    
    generated_sequences = []
    
    for _ in range(num_samples):
        # Create dummy codon labels (will be overwritten during generation)
        dummy_labels = torch.zeros((1, max_length), dtype=torch.long, device=device)
        dummy_labels.fill_(alphabet.padding_idx)
        dummy_labels[0, 0] = alphabet.cls_idx  # Start with CLS token
        
        # Create batch for current generation
        batch_protein_sequence = [protein_seq]
        prompts = [species_prompt]
        
        # Get protein and species embeddings
        batch_protein_tokens, protein_embeddings = model.encode_protein_sequence(batch_protein_sequence)
        
        # Ensure protein embeddings are on the correct device
        protein_embeddings = protein_embeddings.to(device)
        
        # Get species embeddings if species model is available
        if model.species_model is not None:
            species_embeddings = model.extract_species_embedding(prompts)
            # Ensure species embeddings are on the correct device
            species_embeddings = species_embeddings.to(device)
            
            # Combine protein and species embeddings
            species_embeddings = species_embeddings.unsqueeze(1).expand(-1, protein_embeddings.size(1), -1)
            combined_emb = torch.cat((protein_embeddings, species_embeddings), dim=2)
            
            if model.decoder_type == "RNAdecoder":
                protein_embeddings = model.RNA_projector(combined_emb)
            else:
                protein_embeddings = model.prot_species_to_rna(combined_emb)
        
        # Generate sequence step by step
        generated_ids = [alphabet.cls_idx]
        
        for step in range(1, max_length):
            # Create current labels tensor
            current_labels = torch.zeros((1, max_length), dtype=torch.long, device=device)
            current_labels.fill_(alphabet.padding_idx)
            for i, token_id in enumerate(generated_ids):
                if i < max_length:
                    current_labels[0, i] = token_id
            
            # Get logits using the model's forward pass structure
            if model.decoder_type == 'RNAdecoder':
                # Pass through transformer blocks
                hidden_states = protein_embeddings
                for block in model.RNAdecoder.transformer.h:
                    hidden_states = block(hidden_states)
                hidden_states = model.RNAdecoder.transformer.ln_f(hidden_states)
                logits = model.lm_head(hidden_states)
            else:
                logits = model.output_layer(protein_embeddings)
            
            # Get logits for current step (corresponding to protein position)
            if step - 1 < logits.size(1):
                next_token_logits = logits[0, step - 1, :] / temperature
            else:
                break  # Exceeded sequence length
            
            # Apply top-k and top-p filtering
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')
            
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(0, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Sample next token
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1).item()
            
            # Check for end-of-sequence
            if next_token_id == alphabet.eos_idx:
                break
            
            # Check for padding (shouldn't generate padding)
            if next_token_id == alphabet.padding_idx:
                continue
                
            generated_ids.append(next_token_id)
        
        # Convert to codon sequence
        generated_tensor = torch.tensor(generated_ids, device=device)
        codon_sequence = ids_to_codons(generated_tensor, alphabet)
        mrna_sequence = codons_to_mrna(codon_sequence)
        generated_sequences.append(mrna_sequence)
    
    return generated_sequences


def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    model = load_model(args)
    
    # Load input data
    print("Loading protein sequences...")
    proteins = load_protein_sequences(args.protein_fasta)
    
    print("Loading species information...")
    species_df = load_species_info(args.species_csv)
    
    # Generate sequences for each protein-species combination
    results = []
    
    for protein_id, protein_seq in proteins.items():
        print(f"\nProcessing protein: {protein_id}")
        print(f"Protein sequence length: {len(protein_seq)}")
        
        for idx, species_row in species_df.iterrows():
            species_name = species_row.get('organism_name', species_row.get('scientific_name', f'species_{idx}'))
            print(f"  Generating for species: {species_name}")
            
            # Prepare species prompt
            species_prompt = prepare_species_prompt(species_row)
            
            # Generate mRNA sequences
            try:
                mrna_sequences = generate_mrna_sequences(
                    model=model,
                    protein_seq=protein_seq,
                    species_prompt=species_prompt,
                    num_samples=args.num_samples,
                    max_length=args.max_length,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    top_p=args.top_p
                )
                
                # Store results
                for i, mrna_seq in enumerate(mrna_sequences):
                    results.append({
                        'protein_id': protein_id,
                        'protein_sequence': protein_seq,
                        'species_name': species_name,
                        'species_prompt': species_prompt,
                        'sample_id': i,
                        'mrna_sequence': mrna_seq,
                        'mrna_length': len(mrna_seq)
                    })
                
                print(f"    Generated {len(mrna_sequences)} sequences")
                
            except Exception as e:
                print(f"    Error generating sequences: {str(e)}")
                continue
    
    # Save results
    print(f"\nSaving results to {args.output_dir}")
    
    # Save as CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(args.output_dir, 'generated_mrna_sequences.csv'), index=False)
    
    # Save as FASTA files (one per protein-species combination)
    fasta_records = []
    for result in results:
        record_id = f"{result['protein_id']}_{result['species_name']}_sample_{result['sample_id']}"
        description = f"Generated mRNA for {result['protein_id']} in {result['species_name']}"
        record = SeqRecord(Seq(result['mrna_sequence']), id=record_id, description=description)
        fasta_records.append(record)
    
    SeqIO.write(fasta_records, os.path.join(args.output_dir, 'generated_mrna_sequences.fasta'), 'fasta')
    
    # Save summary statistics
    summary = {
        'total_proteins': len(proteins),
        'total_species': len(species_df),
        'total_generated_sequences': len(results),
        'average_mrna_length': float(results_df['mrna_length'].mean()) if len(results) > 0 else 0,
        'min_mrna_length': int(results_df['mrna_length'].min()) if len(results) > 0 else 0,
        'max_mrna_length': int(results_df['mrna_length'].max()) if len(results) > 0 else 0
    }
    
    print(f"summary: {summary}")
    with open(os.path.join(args.output_dir, 'generation_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\nGeneration completed!")
    print(f"Generated {len(results)} total sequences")
    print(f"Results saved in: {args.output_dir}")


if __name__ == "__main__":
    main() 