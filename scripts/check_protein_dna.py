import os
import re

def create_cds_dict(file_path):
    cds_dict = {}
    with open(file_path, 'r') as file:
        protein_id = ''
        sequence = ''
        for line in file:
            if line.startswith('>'):
                if protein_id and sequence:
                    cds_dict[protein_id] = sequence
                # Extract protein ID from the line
                protein_id_match = re.search(r'protein_id=([^ \]]+)', line)
                protein_id = protein_id_match.group(1) if protein_id_match else ''
                sequence = ''
            else:
                sequence += line.strip()
        # Don't forget to save the last entry
        if protein_id and sequence:
            cds_dict[protein_id] = sequence
    return cds_dict

# Now define a function to compare CDS dictionaries from different GCF_ directories
def compare_cds_dicts(data_dir):
    gcf_cds_dicts = {}
    
    # Walk through the directory to read CDS files from each GCF_ folder
    for root, dirs, files in os.walk(data_dir):
        for dir in dirs:
            if dir.startswith('GCF_'):
                cds_file_path = os.path.join(root, dir, 'cds_from_genomic.fna')
                if os.path.isfile(cds_file_path):
                    gcf_cds_dicts[dir] = create_cds_dict(cds_file_path)
    
    
    # Now compare CDS sequences between all GCF_ directories
    comparison_results = {}
    gcf_keys = list(gcf_cds_dicts.keys())
    for i in range(len(gcf_keys)):
        for j in range(i+1, len(gcf_keys)):
            gcf1, gcf2 = gcf_keys[i], gcf_keys[j]
            shared_protein_ids = set(gcf_cds_dicts[gcf1].keys()).intersection(gcf_cds_dicts[gcf2].keys())
            for protein_id in shared_protein_ids:
                cds_seq_gcf1 = gcf_cds_dicts[gcf1][protein_id]
                cds_seq_gcf2 = gcf_cds_dicts[gcf2][protein_id]
                if cds_seq_gcf1 != cds_seq_gcf2:
                    # print(f"protein {protein_id} is different.\n{cds_seq_gcf1} \n vs\n{cds_seq_gcf2}")
                    comparison_key = f"{gcf1} vs {gcf2}"
                    if comparison_key not in comparison_results:
                        comparison_results[comparison_key] = []
                    comparison_results[comparison_key].append(protein_id)
    
    return comparison_results, gcf_cds_dicts

# data_directory = '/Users/zym/Downloads/Okumura_lab/protein2rna/ncbi_dataset/data/test'
data_directory = '/Users/zym/Downloads/Okumura_lab/protein2rna/ncbi_dataset/data/'
comparison_results, gcf_cds_dicts = compare_cds_dicts(data_directory)

# Print the comparison results
# for comparison_key, protein_ids in comparison_results.items():
#     print(f"Differences found in {comparison_key}:")
#     for protein_id in protein_ids:
#         print(f" - Protein ID: {protein_id}")

print(len(comparison_results.keys()))
# print(comparison_results.values())