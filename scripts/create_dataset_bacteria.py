import pandas as pd
import os
import re
import json
import argparse
import random
import gc
import psutil
from collections import defaultdict
from typing import Dict, List, Tuple, Generator, Optional
import logging
from pathlib import Path

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 氨基酸密码子对应表
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

class MemoryMonitor:
    """内存监控器"""
    def __init__(self, threshold: float = 0.8):
        self.threshold = threshold
        
    def check_memory(self) -> Tuple[float, bool]:
        """检查内存使用情况"""
        memory_percent = psutil.virtual_memory().percent / 100.0
        return memory_percent, memory_percent > self.threshold
    
    def force_gc(self):
        """强制垃圾回收"""
        gc.collect()

class StreamingJSONLWriter:
    """流式JSONL写入器"""
    def __init__(self, filepath: str, buffer_size: int = 1000):
        self.filepath = filepath
        self.buffer_size = buffer_size
        self.buffer = []
        self.file = None
        
        # 确保目录存在
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
    def __enter__(self):
        self.file = open(self.filepath, 'w', encoding='utf-8')
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.flush()
        if self.file:
            self.file.close()
    
    def write(self, record: dict):
        """写入一条记录"""
        self.buffer.append(record)
        if len(self.buffer) >= self.buffer_size:
            self.flush()
    
    def flush(self):
        """刷新缓冲区"""
        if self.buffer and self.file:
            for record in self.buffer:
                self.file.write(json.dumps(record, ensure_ascii=False) + '\n')
            self.file.flush()
            self.buffer.clear()

def check_standard_protein_seq(protein_sequence: str) -> bool:
    """检查蛋白质序列是否包含标准氨基酸"""
    return all(aa in amino_acids_codons for aa in protein_sequence)

def check_length(cds_seq_len: int, prot_seq_len: int) -> bool:
    """检查CDS和蛋白质序列长度是否匹配"""
    return cds_seq_len % 3 == 0 and (prot_seq_len + 1) * 3 == cds_seq_len

def check_valid_cds(cds_seq: str) -> bool:
    """检查CDS序列是否只包含有效的DNA碱基"""
    return all(nucleotide in "ATCG" for nucleotide in cds_seq)

def load_taxonomy_info(csv_file_path: str) -> Dict[str, Dict[str, str]]:
    """加载分类学信息 - 优化内存使用"""
    logger.info(f"Loading taxonomic information from: {csv_file_path}")
    
    try:
        # 分块读取CSV文件
        taxonomy_dict = {}
        chunk_size = 10000
        
        for chunk in pd.read_csv(csv_file_path, chunksize=chunk_size):
            for _, row in chunk.iterrows():
                accession = row['Accession']
                taxonomy_dict[accession] = {
                    'superkingdom': str(row.get('superkingdom', '')),
                    'kingdom': str(row.get('kingdom', '')),
                    'phylum': str(row.get('phylum', '')),
                    'class': str(row.get('class', '')),
                    'order': str(row.get('order', '')),
                    'family': str(row.get('family', '')),
                    'genus': str(row.get('genus', '')),
                    'species': str(row.get('species', '')),
                    'organism_name': str(row.get('Organism Name', ''))
                }
            
            # 释放chunk内存
            del chunk
            gc.collect()
        
        logger.info(f"Successfully loaded {len(taxonomy_dict)} taxonomic records")
        return taxonomy_dict
    
    except Exception as e:
        logger.error(f"Failed to load taxonomic information: {e}")
        return {}

def extract_protein_id_from_fasta_header(header: str) -> Optional[str]:
    """从FASTA头部提取protein_id"""
    match = re.search(r'>([A-Z]+_\d+\.\d+)', header)
    return match.group(1) if match else None

def extract_protein_id_from_cds_header(header: str) -> Optional[str]:
    """从CDS FASTA头部提取protein_id"""
    match = re.search(r'\[protein_id=([A-Z]+_\d+\.\d+)\]', header)
    return match.group(1) if match else None

def read_sequences_streaming(file_path: str, extract_id_func) -> Generator[Tuple[str, str], None, None]:
    """流式读取序列文件，避免一次性加载所有数据到内存"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            protein_id = ''
            sequence = ''
            
            for line in file:
                line = line.strip()
                if line.startswith('>'):
                    # 返回前一个序列
                    if protein_id and sequence:
                        yield protein_id, sequence
                    
                    # 开始新序列
                    protein_id = extract_id_func(line)
                    sequence = ''
                else:
                    sequence += line
            
            # 返回最后一个序列
            if protein_id and sequence:
                yield protein_id, sequence
                
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {e}")

def build_sequence_index(file_path: str, extract_id_func) -> Dict[str, int]:
    """构建序列索引，避免加载完整序列到内存"""
    index = {}
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            line_number = 0
            for line in file:
                if line.startswith('>'):
                    protein_id = extract_id_func(line.strip())
                    if protein_id:
                        index[protein_id] = line_number
                line_number += 1
    except Exception as e:
        logger.error(f"Error building index for {file_path}: {e}")
    
    return index

def get_sequence_by_id(file_path: str, protein_id: str, index: Dict[str, int], extract_id_func) -> Optional[str]:
    """根据ID从文件中获取特定序列"""
    if protein_id not in index:
        return None
    
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            
            start_line = index[protein_id]
            if start_line >= len(lines):
                return None
            
            # 从header行开始读取序列
            sequence = ''
            for i in range(start_line + 1, len(lines)):
                line = lines[i].strip()
                if line.startswith('>'):
                    break
                sequence += line
            
            return sequence
    except Exception as e:
        logger.error(f"Error getting sequence for {protein_id}: {e}")
        return None

def extract_gcf_from_filename(filename: str) -> Optional[str]:
    """从文件名中提取GCF编号"""
    match = re.match(r'(GCF_\d+\.\d+)', filename)
    return match.group(1) if match else None

def process_single_organism(gcf_id: str, protein_path: str, cds_path: str, 
                          taxonomy_info: Dict[str, str], memory_monitor: MemoryMonitor) -> Generator[dict, None, None]:
    """处理单个生物体的数据，使用生成器避免内存积累"""
    logger.info(f"Processing {gcf_id}...")
    
    try:
        # 构建CDS序列索引
        cds_index = build_sequence_index(cds_path, extract_protein_id_from_cds_header)
        
        # 流式处理蛋白质序列
        processed_count = 0
        for protein_id, protein_seq in read_sequences_streaming(protein_path, extract_protein_id_from_fasta_header):
            if not protein_id:
                continue
            
            # 获取对应的CDS序列
            cds_seq = get_sequence_by_id(cds_path, protein_id, cds_index, extract_protein_id_from_cds_header)
            if not cds_seq:
                continue
            
            # 数据质量检查
            if (cds_seq.startswith("ATG") and 
                check_valid_cds(cds_seq) and 
                check_standard_protein_seq(protein_seq) and 
                check_length(len(cds_seq), len(protein_seq))):
                
                record = {
                    'gcf_id': gcf_id,
                    'protein_id': protein_id,
                    'protein_sequence': protein_seq,
                    'cds_sequence': cds_seq,
                    'mrna_sequence': cds_seq.replace('T', 'U'),
                    'superkingdom': taxonomy_info['superkingdom'],
                    'kingdom': taxonomy_info['kingdom'],
                    'phylum': taxonomy_info['phylum'],
                    'class': taxonomy_info['class'],
                    'order': taxonomy_info['order'],
                    'family': taxonomy_info['family'],
                    'genus': taxonomy_info['genus'],
                    'species': taxonomy_info['species'],
                    'organism_name': taxonomy_info['organism_name']
                }
                
                yield record
                processed_count += 1
                
                # 定期检查内存
                if processed_count % 100 == 0:
                    memory_percent, should_gc = memory_monitor.check_memory()
                    if should_gc:
                        logger.warning(f"High memory usage ({memory_percent:.1%}), forcing garbage collection")
                        memory_monitor.force_gc()
        
        logger.info(f"Processed {processed_count} valid records for {gcf_id}")
        
    except Exception as e:
        logger.error(f"Error processing {gcf_id}: {e}")

def create_dataset_streaming(data_path: str, output_path: str, taxonomy_csv_path: str, 
                           split_ratio: Optional[Tuple[float, float, float]] = None,
                           max_files: Optional[int] = None):
    """流式创建数据集，避免内存溢出"""
    memory_monitor = MemoryMonitor(threshold=0.8)
    
    # 加载分类学信息
    taxonomy_dict = load_taxonomy_info(taxonomy_csv_path)
    
    # 获取所有的蛋白质文件
    protein_files = [f for f in os.listdir(data_path) if f.endswith('_protein.faa')]
    
    if max_files:
        protein_files = protein_files[:max_files]
        logger.info(f"Limited processing to {max_files} files")
    
    logger.info(f"Found {len(protein_files)} protein files to process")
    
    # 准备输出文件
    if split_ratio:
        # 使用临时文件收集数据，然后分割
        temp_file = os.path.join(output_path, "temp_dataset.jsonl")
        all_records = []
        record_count = 0
        
        # 第一阶段：流式处理并写入临时文件
        with StreamingJSONLWriter(temp_file, buffer_size=500) as writer:
            for i, protein_file in enumerate(protein_files):
                gcf_id = extract_gcf_from_filename(protein_file)
                if not gcf_id:
                    continue
                
                cds_file = f"{gcf_id}_cds_from_genomic.fna"
                protein_path = os.path.join(data_path, protein_file)
                cds_path = os.path.join(data_path, cds_file)
                
                if not os.path.exists(cds_path):
                    logger.warning(f"CDS file does not exist: {cds_file}")
                    continue
                
                # 获取分类学信息
                taxonomy_info = taxonomy_dict.get(gcf_id, {
                    'superkingdom': '', 'kingdom': '', 'phylum': '', 'class': '',
                    'order': '', 'family': '', 'genus': '', 'species': '', 'organism_name': ''
                })
                
                # 流式处理当前生物体
                for record in process_single_organism(gcf_id, protein_path, cds_path, taxonomy_info, memory_monitor):
                    writer.write(record)
                    record_count += 1
                
                # 定期检查内存和进度
                if (i + 1) % 10 == 0:
                    memory_percent, _ = memory_monitor.check_memory()
                    logger.info(f"Processed {i + 1}/{len(protein_files)} files, "
                              f"total records: {record_count}, memory: {memory_percent:.1%}")
                    memory_monitor.force_gc()
        
        logger.info(f"Phase 1 complete: {record_count} total records written to temp file")
        
        # 第二阶段：读取临时文件并分割数据集
        logger.info("Phase 2: Splitting dataset...")
        split_dataset_from_file(temp_file, output_path, split_ratio)
        
        # 清理临时文件
        os.remove(temp_file)
        logger.info("Temporary file cleaned up")
        
    else:
        # 直接写入完整数据集
        output_file = os.path.join(output_path, "bacteria_dataset.jsonl")
        record_count = 0
        
        with StreamingJSONLWriter(output_file, buffer_size=500) as writer:
            for i, protein_file in enumerate(protein_files):
                gcf_id = extract_gcf_from_filename(protein_file)
                if not gcf_id:
                    continue
                
                cds_file = f"{gcf_id}_cds_from_genomic.fna"
                protein_path = os.path.join(data_path, protein_file)
                cds_path = os.path.join(data_path, cds_file)
                
                if not os.path.exists(cds_path):
                    logger.warning(f"CDS file does not exist: {cds_file}")
                    continue
                
                # 获取分类学信息
                taxonomy_info = taxonomy_dict.get(gcf_id, {
                    'superkingdom': '', 'kingdom': '', 'phylum': '', 'class': '',
                    'order': '', 'family': '', 'genus': '', 'species': '', 'organism_name': ''
                })
                
                # 流式处理当前生物体
                for record in process_single_organism(gcf_id, protein_path, cds_path, taxonomy_info, memory_monitor):
                    writer.write(record)
                    record_count += 1
                
                # 定期报告进度
                if (i + 1) % 10 == 0:
                    memory_percent, _ = memory_monitor.check_memory()
                    logger.info(f"Processed {i + 1}/{len(protein_files)} files, "
                              f"total records: {record_count}, memory: {memory_percent:.1%}")
                    memory_monitor.force_gc()
        
        logger.info(f"Complete dataset saved: {record_count} records")

def split_dataset_from_file(input_file: str, output_path: str, split_ratio: Tuple[float, float, float]):
    """从临时文件分割数据集，避免全部加载到内存"""
    train_ratio, valid_ratio, test_ratio = split_ratio
    
    # 第一遍：统计总记录数
    logger.info("Counting total records...")
    total_records = 0
    with open(input_file, 'r', encoding='utf-8') as f:
        for _ in f:
            total_records += 1
    
    logger.info(f"Total records: {total_records}")
    
    # 计算分割点
    train_size = int(total_records * train_ratio)
    valid_size = int(total_records * valid_ratio)
    
    # 创建随机索引
    indices = list(range(total_records))
    random.shuffle(indices)
    
    train_indices = set(indices[:train_size])
    valid_indices = set(indices[train_size:train_size + valid_size])
    test_indices = set(indices[train_size + valid_size:])
    
    # 第二遍：分割数据
    logger.info("Splitting dataset...")
    train_writer = StreamingJSONLWriter(os.path.join(output_path, "train.jsonl"), buffer_size=500)
    valid_writer = StreamingJSONLWriter(os.path.join(output_path, "valid.jsonl"), buffer_size=500)
    test_writer = StreamingJSONLWriter(os.path.join(output_path, "test.jsonl"), buffer_size=500)
    
    try:
        with train_writer, valid_writer, test_writer:
            with open(input_file, 'r', encoding='utf-8') as f:
                for idx, line in enumerate(f):
                    record = json.loads(line.strip())
                    
                    if idx in train_indices:
                        train_writer.write(record)
                    elif idx in valid_indices:
                        valid_writer.write(record)
                    elif idx in test_indices:
                        test_writer.write(record)
                    
                    if (idx + 1) % 10000 == 0:
                        logger.info(f"Split progress: {idx + 1}/{total_records}")
    
    except Exception as e:
        logger.error(f"Error during dataset splitting: {e}")
        raise
    
    train_count = len(train_indices)
    valid_count = len(valid_indices)
    test_count = len(test_indices)
    
    logger.info(f"Dataset split complete: Train {train_count}, Validation {valid_count}, Test {test_count}")

def main():
    parser = argparse.ArgumentParser(description='Create Pro2RNA protein-mRNA dataset (Memory Optimized)')
    parser.add_argument('--data_path', type=str, required=True, 
                       help='Directory path containing protein and CDS files')
    parser.add_argument('--output_path', type=str, required=True,
                       help='Output directory path')
    parser.add_argument('--taxonomy_csv', type=str, required=True,
                       help='CSV file path containing taxonomic information')
    parser.add_argument('--split', action='store_true',
                       help='Whether to split dataset into train/validation/test sets')
    parser.add_argument('--train_ratio', type=float, default=0.7,
                       help='Training set ratio (default: 0.7)')
    parser.add_argument('--valid_ratio', type=float, default=0.15,
                       help='Validation set ratio (default: 0.15)')
    parser.add_argument('--test_ratio', type=float, default=0.15,
                       help='Test set ratio (default: 0.15)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    parser.add_argument('--max_files', type=int, default=None,
                       help='Maximum number of files to process (for testing)')
    parser.add_argument('--memory_threshold', type=float, default=0.8,
                       help='Memory usage threshold for garbage collection (default: 0.8)')
    
    args = parser.parse_args()
    
    # 设置随机种子
    random.seed(args.seed)
    
    # 验证输入参数
    if not os.path.exists(args.data_path):
        logger.error(f"Data path does not exist: {args.data_path}")
        return
    
    if not os.path.exists(args.taxonomy_csv):
        logger.error(f"Taxonomy CSV file does not exist: {args.taxonomy_csv}")
        return
    
    # 验证比例
    if args.split:
        total_ratio = args.train_ratio + args.valid_ratio + args.test_ratio
        if abs(total_ratio - 1.0) > 1e-6:
            logger.error(f"train/validation/test ratios should sum to 1.0, got {total_ratio}")
            return
        
        split_ratio = (args.train_ratio, args.valid_ratio, args.test_ratio)
    else:
        split_ratio = None
    
    # 创建输出目录
    os.makedirs(args.output_path, exist_ok=True)
    
    # 记录系统信息
    memory_info = psutil.virtual_memory()
    logger.info(f"System memory: {memory_info.total / (1024**3):.1f} GB total, "
               f"{memory_info.available / (1024**3):.1f} GB available")
    
    # 创建数据集
    logger.info("Starting dataset creation with memory optimization...")
    create_dataset_streaming(args.data_path, args.output_path, args.taxonomy_csv, 
                            split_ratio, args.max_files)
    
    logger.info("Dataset creation completed!")

if __name__ == "__main__":
    main()

# Usage example:
# python scripts/create_dataset_bacteria.py --data_path data/bacteria --output_path data/output --taxonomy_csv data/filtered_bacteria_species_updated_final.csv --split --max_files 100