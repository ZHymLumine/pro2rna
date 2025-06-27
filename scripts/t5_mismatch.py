import csv

def translate_rna_to_protein(rna_sequence):
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

    protein = []
    for i in range(0, len(rna_sequence), 3):
        codon = rna_sequence[i:i + 3]
        if len(codon) == 3:
            protein.append(codon_table.get(codon, '-'))  # Use '-' for invalid codons

    return ''.join(protein)

def rna_to_protein_mismatch(csv_file_path, output_file_path):
    mismatch_count = 0
    total_sequences = 0

    with open(csv_file_path, "r", encoding="utf-8") as file:
        csv_reader = csv.DictReader(file)

        with open(output_file_path, "w", encoding="utf-8", newline="") as output_file:
            fieldnames = ["True RNA Sequence", "Predicted RNA Sequence", "True Protein Sequence", "Predicted Protein Sequence", "Mismatch"]
            csv_writer = csv.DictWriter(output_file, fieldnames=fieldnames)
            csv_writer.writeheader()

            invalid_predicted_lengths = []
            for row in csv_reader:
                true_rna = row["True RNA Sequence"].replace(" ", "")  # Remove spaces if present
                predicted_rna = row["Predicted RNA Sequence"].replace(" ", "")

                if len(true_rna) % 3 != 0 or len(predicted_rna) % 3 != 0:
                    if len(predicted_rna) % 3 != 0:
                        invalid_predicted_lengths.append(len(predicted_rna))
                    print(f"Invalid RNA sequence length detected. True RNA: {len(true_rna)}, Predicted RNA: {len(predicted_rna)}")
                    continue

                # Convert RNA to protein sequences
                true_protein = translate_rna_to_protein(true_rna)
                predicted_protein = translate_rna_to_protein(predicted_rna)

                # Check for mismatch
                mismatch = true_protein != predicted_protein
                if mismatch:
                    mismatch_count += 1

                total_sequences += 1

                # Write to output file
                # if mismatch:
                csv_writer.writerow({
                    "True Protein Sequence": true_protein,
                    "Predicted Protein Sequence": predicted_protein,
                    "Mismatch": mismatch
                })
            if invalid_predicted_lengths:
                min_length = min(invalid_predicted_lengths)
                max_length = max(invalid_predicted_lengths)
                avg_length = sum(invalid_predicted_lengths) / len(invalid_predicted_lengths)
                print(f"Statistics for invalid predicted RNA lengths:")
                print(f" - Minimum length: {min_length}")
                print(f" - Maximum length: {max_length}")
                print(f" - Average length: {avg_length:.2f}")
                print(f" - Total invalid sequences: {len(invalid_predicted_lengths)}")
            else:
                print("No invalid predicted RNA lengths detected.")
    # Calculate mismatch ratio
    mismatch_ratio = mismatch_count / total_sequences if total_sequences > 0 else 0

    print(f"Total Sequences: {total_sequences}")
    print(f"Mismatch Count: {mismatch_count}")
    print(f"Mismatch Ratio: {mismatch_ratio:.2%}")

csv_file_path = "/Users/zym/Downloads/Okumura_lab/protein2rna/scripts/predictions_length_label.csv"  
output_file_path = "mismatch_2_token.csv"
rna_to_protein_mismatch(csv_file_path, output_file_path)