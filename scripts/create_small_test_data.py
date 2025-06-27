import os

test_folder_path = '/Users/zym/Downloads/Okumura_lab/protein2rna/ncbi_dataset/data/test'

for folder_name in os.listdir(test_folder_path):
    if folder_name.startswith('GCF_'):
        folder_path = os.path.join(test_folder_path, folder_name)
        protein_faa_path = os.path.join(folder_path, 'protein.faa')
        
        # 检查protein.faa文件是否存在
        if os.path.exists(protein_faa_path):
            with open(protein_faa_path, 'r') as file:
                lines = file.readlines()
                
            count = 0 
            top500_records = []
            for line in lines:
                if line.startswith('>'):
                    count += 1
                    if count > 500:
                        break
                top500_records.append(line)
            
            new_file_path = os.path.join(folder_path, 'protein_top500.faa')
            with open(new_file_path, 'w') as new_file:
                new_file.writelines(top500_records)

print("处理完成。")
