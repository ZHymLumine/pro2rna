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

def sample_codon(amino_acid):
    """
    Given an amino acid (single-letter code), sample a codon that encodes it with equal probability.
    """
    # Check if the amino acid is valid
    if amino_acid not in amino_acids_codons:
        return "TGA"
    
    # Randomly select a codon for the given amino acid
    codons = amino_acids_codons[amino_acid]
    return random.choice(codons)


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
            cds_seq = entry["cds_sequence"]
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


# 读取protein.faa文件提取蛋白质ID和氨基酸序列
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
    return protein_sequences

# 读取cds_from_genomic.fna文件提取蛋白质ID和密码子序列
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
    return codon_sequences

def calculate_accuracy(sampled_seq, real_seq):
    """计算采样序列与真实序列的准确率。"""
    correct = sum(1 for i in range(0, len(real_seq), 3) if sampled_seq[i:i+3] == real_seq[i:i+3])
    total = len(real_seq) // 3
    return correct / total if total > 0 else 0


def check_standard_protein_seq(protein_sequence):
    for aa in protein_sequence:
        if aa not in amino_acids_codons:
            return False
    return True
        
def evaluate_accuracy(data_dir, num_experiments=10):
    """评估给定目录下所有GCF_文件夹的准确率, 并返回结果字典。"""
    species_accuracies = {}  # Store accuracies for each species
    
    for species in os.listdir(data_dir):
        if species.startswith('GCF_'):
            protein_file = os.path.join(data_dir, species, 'protein.faa')
            cds_file = os.path.join(data_dir, species, 'cds_from_genomic.fna')
            if os.path.exists(protein_file) and os.path.exists(cds_file):
                protein_seqs = read_protein_sequences(protein_file)
                cds_seqs = read_codon_sequences(cds_file)
                for _ in range(num_experiments):
                    print(f"Species: {species}, No: {_ + 1}")
                    accuracies = []
                    for protein_id, amino_acid_seq in protein_seqs.items():
                        # sampled_codon_seq = ''.join(sample_codon(aa) for aa in amino_acid_seq) + random.choice(amino_acids_codons['*'])   # add terminate codon
                        # sampled_codon_seq = ''.join(sample_codon(aa) for aa in amino_acid_seq)
                        sampled_codon_seq = ''.join(random.choice(all_codons) for _ in amino_acid_seq)  # Totally random sample codons w/o considering amino acid information
                        
                        real_codon_seq = cds_seqs.get(protein_id, '')[:-3]
                        if real_codon_seq.startswith('ATG') and check_standard_protein_seq(amino_acid_seq):
                            # print(f"amino_acid_seq: {amino_acid_seq}, real_codon_seq: {real_codon_seq}")
                            accuracy = calculate_accuracy(sampled_codon_seq, real_codon_seq)
                            accuracies.append(accuracy)
                    # Add accuracies to the list for each species
                    if species not in species_accuracies:
                        species_accuracies[species] = []
                    species_accuracies[species].extend(accuracies)

    # Calculate average and standard deviation of accuracies for each species
    species_stats = {species: {'avg': mean(acc), 'std': stdev(acc) if len(acc) > 1 else 0}
                     for species, acc in species_accuracies.items()}

    # Calculate overall average and standard deviation
    all_accuracies = [acc for accuracies in species_accuracies.values() for acc in accuracies]
    overall_avg = mean(all_accuracies)
    overall_std = stdev(all_accuracies) if len(all_accuracies) > 1 else 0

    return species_stats, overall_avg, overall_std

def evaluate_accuracy_from_json(json_file, num_experiments=10, codon_frequencies=None):
    species_accuracies = {}
    with open(json_file, 'r') as file:
        data = json.load(file)
        for entry in data:
            organism_name = entry["Organism Scientific Name"]
            species = entry['Assembly Accession']
            # protein_seqs = entry["protein_seq"]
            # cds_seqs = entry["cds_seq"]
            protein_seqs = entry["prot_sequence"]
            cds_seqs = entry["cds_sequence"]
            accuracies = []
            for _ in range(num_experiments):
                print(f"Species: {species}, No: {_ + 1}")
                accuracies = []
                # for protein_id, amino_acid_seq in protein_seqs.items():
                #     # sampled_codon_seq = ''.join(sample_codon(aa) for aa in amino_acid_seq) + random.choice(amino_acids_codons['*'])
                #     sampled_codon_seq = ''.join(sample_codon_weighted(aa) for aa in amino_acid_seq)
                #     real_codon_seq = cds_seqs.get(protein_id, '')[:-3]
                #     if real_codon_seq.startswith('ATG') and check_standard_protein_seq(amino_acid_seq):
                #         # print(f"amino_acid_seq: {amino_acid_seq}, real_codon_seq: {real_codon_seq}")
                #         accuracy = calculate_accuracy(sampled_codon_seq, real_codon_seq)
                #         accuracies.append(accuracy)
                sampled_codon_seq = ''.join(sample_codon_weighted(aa, codon_frequencies) for aa in protein_seqs)
                real_codon_seq = cds_seqs[:-3]
                if real_codon_seq.startswith('ATG') and check_standard_protein_seq(protein_seqs):
                    # print(f"amino_acid_seq: {amino_acid_seq}, real_codon_seq: {real_codon_seq}")
                    accuracy = calculate_accuracy(sampled_codon_seq, real_codon_seq)
                    accuracies.append(accuracy)
                # Add accuracies to the list for each species
                if species not in species_accuracies:
                    species_accuracies[species] = []
                species_accuracies[species].extend(accuracies)
           
    species_stats = {species: {'avg': mean(acc), 'std': stdev(acc) if len(acc) > 1 else 0}
                     for species, acc in species_accuracies.items()}

    # Calculate overall average and standard deviation
    all_accuracies = [acc for accuracies in species_accuracies.values() for acc in accuracies]
    overall_avg = mean(all_accuracies)
    overall_std = stdev(all_accuracies) if len(all_accuracies) > 1 else 0

    return species_stats, overall_avg, overall_std


if __name__ == "__main__":
    

    data_dirs = ['test_flat.json']
    data_path = "/Users/zym/Downloads/Okumura_lab/protein2rna/ncbi_dataset/data"

    train_data_path = os.path.join(data_path, "train_flat.json")
    codon_frequencies = calculate_codon_frequencies(train_data_path)

    print("频率计算完成")
    # Evaluate each directory and store results
    overall_results = {}
    for data_dir in data_dirs:
        dir_path = os.path.join(data_path, data_dir)
        species_stats, overall_avg, overall_std = evaluate_accuracy_from_json(dir_path, codon_frequencies=codon_frequencies)
        overall_results[data_dir] = {
            'species_stats': species_stats,
            'overall_avg': overall_avg,
            'overall_std': overall_std
        }

    with open('test_accuracy_results_statistics.json', 'w') as json_file:
        json.dump(overall_results, json_file, indent=4)

    print("准确率评估完成")