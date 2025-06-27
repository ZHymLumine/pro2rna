import random
import torch

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

RNA_CODONS = {
    'A': ['GCU', 'GCC', 'GCA', 'GCG'], # Alanine
    'R': ['CGU', 'CGC', 'CGA', 'CGG', 'AGA', 'AGG'], # Arginine
    'N': ['AAU', 'AAC'], # Asparagine
    'D': ['GAU', 'GAC'], # Aspartic acid
    'C': ['UGU', 'UGC'], # Cysteine
    'E': ['GAA', 'GAG'], # Glutamic acid
    'Q': ['CAA', 'CAG'], # Glutamine
    'G': ['GGU', 'GGC', 'GGA', 'GGG'], # Glycine
    'H': ['CAU', 'CAC'], # Histidine
    'I': ['AUU', 'AUC', 'AUA'], # Isoleucine
    'L': ['UUA', 'UUG', 'CUU', 'CUC', 'CUA', 'CUG'], # Leucine
    'K': ['AAA', 'AAG'], # Lysine
    'M': ['AUG'], # Methionine
    'F': ['UUU', 'UUC'], # Phenylalanine
    'P': ['CCU', 'CCC', 'CCA', 'CCG'], # Proline
    'S': ['UCU', 'UCC', 'UCA', 'UCG', 'AGU', 'AGC'], # Serine
    'T': ['ACU', 'ACC', 'ACA', 'ACG'], # Threonine
    'W': ['UGG'], # Tryptophan
    'Y': ['UAU', 'UAC'], # Tyrosine
    'V': ['GUU', 'GUC', 'GUA', 'GUG'], # Valine
    '*': ['UAA', 'UAG', 'UGA'] # Stop codon
}


def check_standard_protein_seq(protein_sequence):
    for aa in protein_sequence:
        if aa not in amino_acids_codons:
            return False
    return True


def sample_codon_id_seqs(batch_protein_ids, esm_alphabet, codon_alphabet):
    """
    Given a batch of protein sequence, randomly sample a batch codon sequence id that encodes it with equal probability.
    """
    protein_idx_to_tok = {i: tok for i, tok in enumerate(esm_alphabet.all_toks)}
    batch_codon_seqs = []
    for protein_seq in batch_protein_ids:
        seq_codon = []
        for protein_id in protein_seq:
            amino_acid = protein_idx_to_tok.get(protein_id.item())
            if amino_acid in ['<cls>', '<pad>']:
                seq_codon.append(codon_alphabet.tok_to_idx.get(amino_acid)) # Directly append the special token
            elif amino_acid == '<eos>':
                seq_codon.append(codon_alphabet.tok_to_idx.get(random.choice(RNA_CODONS['*'])))  # Randomly choose a stop codon
            else: 
                # Standard amino acid, randomly select a codon
                codons = RNA_CODONS.get(amino_acid, "")
                if codons:
                    seq_codon.append(codon_alphabet.tok_to_idx.get(random.choice(codons)))
                else:
                    seq_codon.append(codon_alphabet.tok_to_idx.get('<pad>'))  # Placeholder for invalid amino acids
        batch_codon_seqs.append(seq_codon)
    batch_codon_seqs = torch.tensor(batch_codon_seqs)
    return batch_codon_seqs