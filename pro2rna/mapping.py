import random

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

def sample_codon(amino_acid):
    """
    Given an amino acid (single-letter code), sample a codon that encodes it with equal probability.
    """
    # Check if the amino acid is valid
    if amino_acid not in amino_acids_codons:
        return None
    
    # Randomly select a codon for the given amino acid
    codons = amino_acids_codons[amino_acid]
    return random.choice(codons)

# Example: Sample a codon for Alanine (A)
sampled_codon = sample_codon('A')
print(sampled_codon)