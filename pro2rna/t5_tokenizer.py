# import argparse
# from transformers import AutoTokenizer

# # 命令行参数
# parser = argparse.ArgumentParser(description="T5 Tokenizer training script.")
# parser.add_argument("--base_tokenizer", type=str, default="t5-base", help="Base tokenizer.")
# parser.add_argument("--txt_file_path", type=str, required=True, help="Path to the text file for training.")
# parser.add_argument("--input_tokenizer_path", type=str, required=True, help="Path to save the input tokenizer.")
# parser.add_argument("--label_tokenizer_path", type=str, required=True, help="Path to save the label tokenizer.")
# parser.add_argument("--push_to_hub", action='store_true', help="Whether to push the tokenizer to Hugging Face's model hub.")
# args = parser.parse_args()

# print("Base Tokenizer:", args.base_tokenizer)
# print("Text File Path:", args.txt_file_path)
# print("Input Tokenizer Path:", args.input_tokenizer_path)
# print("Label Tokenizer Path:", args.label_tokenizer_path)
# print("Push to Hub:", args.push_to_hub)

# # 加载基础 tokenizer
# input_tokenizer = AutoTokenizer.from_pretrained(args.base_tokenizer)
# label_tokenizer = AutoTokenizer.from_pretrained(args.base_tokenizer)

# # 提取 special tokens
# special_tokens = list(input_tokenizer.special_tokens_map_extended.values())

# # ------------------------------
# # 构建 Input Tokenizer (氨基酸 + taxonomy)
# # ------------------------------
# taxonomy_tokens = set()
# with open(args.txt_file_path, "r", encoding="utf-8") as file:
#     for line in file:
#         # 提取 taxonomy 信息，跳过 mRNA 列
#         species_info = line.strip().split(',')[1:]
#         formatted_species_info = {f"<{tax}>" for tax in species_info}
#         taxonomy_tokens.update(formatted_species_info)

# amino_acids = ["A", "R", "N", "D", "C", "Q", "E", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"]

# # 清空原有词汇表，仅保留 special tokens
# input_tokenizer_vocab = special_tokens.copy()
# input_tokenizer_vocab += list(taxonomy_tokens) + amino_acids
# # input_tokenizer._tokenizer = input_tokenizer._tokenizer.train_new_from_iterator(
# #     input_tokenizer_vocab, len(input_tokenizer_vocab)
# # )
# print(f"Added {len(input_tokenizer_vocab)} tokens to the input tokenizer.")

# # 保存 Input Tokenizer
# input_tokenizer.save_pretrained(args.input_tokenizer_path)
# if args.push_to_hub:
#     input_tokenizer.push_to_hub("ZYMScott/CodonT5_input")

# # ------------------------------
# # 构建 Label Tokenizer (RNA Codons)
# # ------------------------------
# rna_codons = [
#     "UUU", "UUC", "UUA", "UUG", "CUU", "CUC", "CUA", "CUG", "AUU", "AUC", "AUA", "AUG",
#     "GUU", "GUC", "GUA", "GUG", "UCU", "UCC", "UCA", "UCG", "CCU", "CCC", "CCA", "CCG",
#     "ACU", "ACC", "ACA", "ACG", "GCU", "GCC", "GCA", "GCG", "UAU", "UAC", "UAA", "UAG",
#     "CAU", "CAC", "CAA", "CAG", "AAU", "AAC", "AAA", "AAG", "GAU", "GAC", "GAA", "GAG",
#     "UGU", "UGC", "UGA", "UGG", "CGU", "CGC", "CGA", "CGG", "AGU", "AGC", "AGA", "AGG",
#     "GGU", "GGC", "GGA", "GGG"
# ]

# # 清空原有词汇表，仅保留 special tokens
# label_tokenizer_vocab = special_tokens.copy()
# label_tokenizer_vocab += rna_codons
# # label_tokenizer._tokenizer = label_tokenizer._tokenizer.train_new_from_iterator(
# #     label_tokenizer_vocab, len(label_tokenizer_vocab)
# # )
# print(f"Added {len(label_tokenizer_vocab)} tokens to the label tokenizer.")

# # 保存 Label Tokenizer
# label_tokenizer.save_pretrained(args.label_tokenizer_path)
# if args.push_to_hub:
#     label_tokenizer.push_to_hub("ZYMScott/CodonT5_label")



import argparse
import os
from tqdm import tqdm
from transformers import AutoTokenizer

# 命令行参数
parser = argparse.ArgumentParser(description="T5 Tokenizer training script.")
parser.add_argument("--base_tokenizer", type=str, default="t5-base", help="Base tokenizer.")
parser.add_argument("--txt_file_path", type=str, required=True, help="Path to the text file for training.")
parser.add_argument("--input_tokenizer_path", type=str, required=True, help="Path to save the input tokenizer.")
parser.add_argument("--label_tokenizer_path", type=str, required=True, help="Path to save the label tokenizer.")
parser.add_argument("--push_to_hub", action='store_true', help="Whether to push the tokenizer to Hugging Face's model hub.")
args = parser.parse_args()

print("Base Tokenizer:", args.base_tokenizer)
print("Text File Path:", args.txt_file_path)
print("Input Tokenizer Path:", args.input_tokenizer_path)
print("Label Tokenizer Path:", args.label_tokenizer_path)
print("Push to Hub:", args.push_to_hub)

# 加载基础 tokenizer
input_tokenizer = AutoTokenizer.from_pretrained(args.base_tokenizer)
label_tokenizer = AutoTokenizer.from_pretrained(args.base_tokenizer)

# ------------------------------
# 构建 Input Tokenizer (氨基酸 + taxonomy)
# ------------------------------
# 初始化 taxonomy 的 token 集合
taxonomy_tokens = set()
with open(args.txt_file_path, "r", encoding="utf-8") as file:
    for line in file:
        species_info = line.strip().split(',')[1:]
        formatted_species_info = {f"<{tax}>" for tax in species_info}
        taxonomy_tokens.update(formatted_species_info)

amino_acids = ["A", "R", "N", "D", "C", "Q", "E", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"]

# 合并 amino_acids 和 taxonomy tokens
input_tokens = list(taxonomy_tokens) + amino_acids

num_added_input_tokens = input_tokenizer.add_tokens(input_tokens)
print(f"Added {num_added_input_tokens} new tokens to the input tokenizer.")

input_tokenizer.save_pretrained(args.input_tokenizer_path)
if args.push_to_hub:
    input_tokenizer.push_to_hub("ZYMScott/CodonT5_input")

# ------------------------------
# 构建 Label Tokenizer (RNA Codons)
# ------------------------------
rna_codons = [
    "UUU", "UUC", "UUA", "UUG", "CUU", "CUC", "CUA", "CUG", "AUU", "AUC", "AUA", "AUG",
    "GUU", "GUC", "GUA", "GUG", "UCU", "UCC", "UCA", "UCG", "CCU", "CCC", "CCA", "CCG",
    "ACU", "ACC", "ACA", "ACG", "GCU", "GCC", "GCA", "GCG", "UAU", "UAC", "UAA", "UAG",
    "CAU", "CAC", "CAA", "CAG", "AAU", "AAC", "AAA", "AAG", "GAU", "GAC", "GAA", "GAG",
    "UGU", "UGC", "UGA", "UGG", "CGU", "CGC", "CGA", "CGG", "AGU", "AGC", "AGA", "AGG",
    "GGU", "GGC", "GGA", "GGG"
]

num_added_label_tokens = label_tokenizer.add_tokens(rna_codons)
print(f"Added {num_added_label_tokens} new tokens to the label tokenizer.")

label_tokenizer.save_pretrained(args.label_tokenizer_path)
if args.push_to_hub:
    label_tokenizer.push_to_hub("ZYMScott/CodonT5_label")

# python t5_tokenizer.py --txt_file_path /raid_elmo/home/lr/zym/mRNAdesigner/data/rna_sequence.txt --input_tokenizer_path ./CodonT5_input_tokenizer --label_tokenizer_path ./CodonT5_label_tokenizer --push_to_hub