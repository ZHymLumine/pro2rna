import random
import os
import json
import re
from statistics import mean, stdev

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


def sample_codon_weighted(amino_acid, codon_frequencies):
    """
    Given an amino acid (single-letter code) and codon frequencies, sample a codon that encodes it based on the frequencies.
    """
    if amino_acid not in codon_frequencies:
        return "TGA"  # Return a stop codon if the amino acid is not recognized
    
    codons = list(codon_frequencies[amino_acid].keys())
    weights = list(codon_frequencies[amino_acid].values())
    return random.choices(codons, weights=weights)[0]


def calculate_codon_frequencies(train_data_path):
    """
    Calculate the frequencies of each codon for each amino acid in the training set.
    """
    codon_frequencies = {aa: {} for aa in amino_acids_codons}
    with open(train_data_path, 'r') as file:
        train_data = json.load(file)
        for entry in train_data:
            cds_seq = entry["cds_seq"]
            for i in range(0, len(cds_seq), 3):
                codon = cds_seq[i:i+3]
                for aa, codons in amino_acids_codons.items():
                    if codon in codons:
                        if codon in codon_frequencies[aa]:
                            codon_frequencies[aa][codon] += 1
                        else:
                            codon_frequencies[aa][codon] = 1
                        break

    # Normalize frequencies to get probabilities
    for aa, codons in codon_frequencies.items():
        total = sum(codons.values())
        for codon in codons:
            codon_frequencies[aa][codon] /= total

    return codon_frequencies
