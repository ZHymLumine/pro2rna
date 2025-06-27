import os
import shutil
from sklearn.model_selection import train_test_split
import argparse


def copy_folders(data_root_path, folders, destination):
    for folder in folders:
        src_path = os.path.join(data_root_path, folder)
        dst_path = os.path.join(data_root_path, destination, folder)
        shutil.copytree(src_path, dst_path)

def main(args):
    data_root_path = args.base_path

    species_folders = [d for d in os.listdir(data_root_path) if os.path.isdir(os.path.join(data_root_path, d)) and d.startswith('GCF_')]

    train_val, test = train_test_split(species_folders, test_size=0.2, random_state=42)
    train, valid = train_test_split(train_val, test_size=0.125, random_state=42) # 0.22约等于原始总量的20%


    for folder in ['train', 'valid', 'test']:
        os.makedirs(folder, exist_ok=True)
    
    copy_folders(data_root_path, train, 'train')
    copy_folders(data_root_path, valid, 'valid')
    copy_folders(data_root_path, test, 'test')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_path", type=str, required=True)
    args = parser.parse_args()
    main(args)

    print("Split done")

# python split_dataset.py --base_path /Users/zym/Downloads/Okumura_lab/protein2rna/ncbi_dataset/data
