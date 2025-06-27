import os
import random
from collections import defaultdict
import pandas as pd

# 假设test_dir是包含所有GCF_文件夹的目录路径
test_dir = '/Users/zym/Downloads/Okumura_lab/protein2rna/ncbi_dataset/data/test'

# 初始化字典，用于存储每个序列出现的次数
sequence_counts = defaultdict(int)

# 初始化字典，用于存储序列对应的GCF编号
sequence_to_gcf = defaultdict(set)

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


global_dict = {}

# 读取每个GCF_文件夹中的protein.faa文件
for folder in os.listdir(test_dir):
    if folder.startswith('GCF_'):
        file_path = os.path.join(test_dir, folder, 'protein.faa')
        protein_sequences = read_protein_sequences(file_path)
        # print(protein_sequences)
        # print(type(protein_sequences))
        for protein_id, protein_seq in protein_sequences.items():
            if protein_id not in global_dict:
                global_dict[protein_id] = []
            global_dict[protein_id].append((protein_seq, folder))
        # with open(file_path, 'r') as file:
        #     sequences = file.read().split('>')[1:]  # 跳过第一个空元素
        #     print(sequences)
        # for sequence in sequences:
        #     seq_id = sequence.split('\n', 1)[0]  # 提取序列ID
        #     sequence_counts[seq_id] += 1
        #     sequence_to_gcf[seq_id].add(folder)

# print(global_dict)
common_protein_ids = [protein_id for protein_id, entries in global_dict.items() if len(entries) == 11]
print(common_protein_ids)
# 找出所有物种共有的蛋白质序列
total_gcf_folders = sum(folder.startswith('GCF_') for folder in os.listdir(test_dir))
common_sequences = [seq for seq, count in sequence_counts.items() if count == total_gcf_folders]

# 如果共有的蛋白质少于10个，则选择所有，否则随机选择10个
selected_sequences = common_sequences if len(common_sequences) <= 10 else random.sample(common_sequences, 10)

# 准备数据以保存到CSV
data = {'Assembly Accession': [], 'protein_sequence': []}
# for seq in selected_sequences:
    # for gcf in sequence_to_gcf[seq]:
        # data['Assembly Accession'].append(gcf)
        # data['protein_sequence'].append(seq)

# 保存到CSV文件
# df = pd.DataFrame(data)
# df.to_csv('/path/to/output.csv', index=False)
