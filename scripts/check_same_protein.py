import os
import argparse

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

def read_codons(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        codons = []
        for line in lines:
            parts = line.strip().split('\t')
            codon_seq = parts[1].replace("<cls> ", "").replace(" <eos>", "").strip().split(' ')
            codons.append(codon_seq)
    file.close()
    return codons

def codon_to_protein(codons):
    protein = ''
    for codon in codons:
        if codon in codon_table:
            protein += codon_table[codon]
    return protein

def compare_proteins(real_codons, predicted_codons, args):
    mismatch_count = 0
    mismatch_ratios = []
    mismatch_protein_file_path = os.path.join(args.output_dir, 'mismatch_protein_seqs.txt')
    mismatch_protein_seq_file = open(mismatch_protein_file_path, 'w')
    protein_cnt = 0
    total_proteins = len(real_codons)

    for i, (real, predicted) in enumerate(zip(real_codons, predicted_codons)):
        protein_cnt += 1
        real_protein = codon_to_protein(real)
        predicted_protein = codon_to_protein(predicted)
        if real_protein.endswith('*'):
            real_protein = real_protein[:-1]
        if predicted_protein.endswith('*'):
            predicted_protein = predicted_protein[:-1]
            
        if real_protein != predicted_protein:
            mismatch_count += 1
            mismatches = sum(1 for a, b in zip(real_protein, predicted_protein) if a != b)
            total = max(len(real_protein), len(predicted_protein))
            mismatch_ratio = mismatches / total
            mismatch_ratios.append(mismatch_ratio)
            mismatch_protein_seq_file.write(f"Real: {real_protein}\nPredicted: {predicted_protein}\nMismatch counts:{mismatches}\nMismatch Ratio: {mismatch_ratio}\n")

        # Update progress
        percent_complete = (i + 1) / total_proteins * 100
        print(f'Processing {i+1}/{total_proteins} proteins ({percent_complete:.2f}%) complete', end='\r')
    
    print()  # To ensure the next output is on a new line
    return mismatch_count, mismatch_ratios, protein_cnt


def main(args):
    real_codons = read_codons(args.real_seq_path)
    predicted_codons = read_codons(args.predicted_seq_path)
    print(f"real: {len(real_codons)}")
    print(f"predicted: {len(predicted_codons)}")
    mismatch_count, mismatch_ratios, total_cnt = compare_proteins(real_codons, predicted_codons, args)
    print(f"Total proteins with mismatches: {mismatch_count}")
    print(f"Total proteins counts: {total_cnt}")
    print(f"Mismatch Ratio: {mismatch_count / total_cnt}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--real_seq_path", type=str, required=True)
    parser.add_argument("--predicted_seq_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()
    main(args)


# python check_same_protein.py --real_seq_path /Users/zym/Downloads/Okumura_lab/protein2rna/results/withRNA/real_sequences_withRNAmodel.txt --predicted_seq_path /Users/zym/Downloads/Okumura_lab/protein2rna/results/withRNA/predicted_sequences_withRNAmodel.txt --output_dir /Users/zym/Downloads/Okumura_lab/protein2rna/results/withRNA

# python check_same_protein.py --real_seq_path /Users/zym/Downloads/Okumura_lab/protein2rna/results/withoutRNA/real_sequences_withoutRNAmodel.txt --predicted_seq_path /Users/zym/Downloads/Okumura_lab/protein2rna/results/withoutRNA/predicted_sequences_withoutRNAmodel.txt --output_dir /Users/zym/Downloads/Okumura_lab/protein2rna/results/withoutRNA