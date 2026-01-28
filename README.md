# Pro2RNA: Multimodal Framework for Organism-Specific mRNA Generation

Pro2RNA is a cutting-edge multimodal framework that generates organism-specific mRNA sequences by combining taxonomic context, protein semantics, and pre-trained RNA language models. The framework leverages **SciBERT** for taxonomy encoding, **ESM-2** for protein encoding, and **pre-trained RNA/mRNA GPT models** for sequence generation.

## 🚀 Key Features

- **Multimodal Architecture**: Integrates taxonomic and protein information for context-aware generation
- **Pre-trained RNA Decoder**: Utilizes GenRNA, mRNA GPT, or similar pre-trained models with LoRA fine-tuning
- **Parameter-Efficient Training**: Uses LoRA adapters for all large language models
- **Organism-Specific Codon Optimization**: Generates mRNA sequences optimized for specific organisms
- **Flexible Integration**: Easy to integrate with existing RNA language models

## 🏗️ Architecture

### Core Components

1. **TaxonomyEncoder** (SciBERT + LoRA)
   - Encodes natural language taxonomy prompts
   - Format: "The organism {name} belongs to order {order}, family {family}, genus {genus}, species {species}"
   - Uses parameter-efficient LoRA fine-tuning

2. **ProteinEncoder** (ESM-2 650M + LoRA)
   - Encodes amino acid sequences into protein semantic embeddings
   - Leverages ESM-2's understanding of protein structure and function
   - LoRA adaptation for task-specific tuning

3. **Fusion**
   - Fuses taxonomic and protein embeddings
   - Enables conditional dependencies for organism-specific generation
   - Output: Rich multimodal context embeddings

4. **mRNADecoder** (Pre-trained RNA/mRNA GPT + LoRA)
   - **NEW**: Uses pre-trained GenRNA or mRNA GPT models
   - Direct embedding feeding into transformer layers
   - LoRA fine-tuning for organism-specific adaptation
   - Autoregressive mRNA sequence generation

### Architecture Flow

```
Taxonomy Prompt → SciBERT+LoRA → Taxonomy Embeddings
                                      ↓
                                  Fused Context
                                      ↑
Protein Sequence → ESM-2+LoRA → Protein Embeddings
                                      ↓
Fused Context → Context Projector → Pre-trained mRNA GPT+LoRA → mRNA Sequence
```

## 📦 Installation

### Requirements


### Installation Steps

```bash
# Clone the repository
git clone https://github.com/your-repo/pro2rna.git
cd pro2rna

conda create -n pro2rna python=3.10 -y
# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

## Create dataset
```bash
python scripts/create_dataset_bacteria.py --data_path data/bacteria --output_path data/output --taxonomy_csv data/filtered_bacteria_species_updated_final.csv --split --max_files 100

python build_dataset.py --data_path /home/yzhang/research/pro2rna/data/output --out_dir /home/yzhang/research/pro2rna/data/build/

```


## 🏃‍♂️ Training

### Training with Pre-trained mRNA Decoder

```bash
num_gpus=1
torchrun --nnodes 1 --nproc_per_node ${num_gpus}  pro2rna/training.py \
    --dataset_path /path/to/data \
    --output_dir /path/to/output \
    --esm_name_or_path "esm2_t12_35M_UR50D" \
    --species_model "scibert" \
    --num_train_epochs 20 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --evaluation_strategy "epoch" \
    --gradient_accumulation_steps 8 \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 1 \
    --learning_rate 6e-4 \
    --weight_decay 0. \
    --lr_scheduler_type "cosine" \
    --dataloader_num_workers 6 \
    --logging_steps 1 \
    --report_to wandb \
    --decoder_type "RNAdecoder" \
    --RNA_config_path /path/to/config.json \
    --decoder_path /path/to/decoder.pt \
```

### Training Configuration Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `mrna_pretrained_model` | Path to pre-trained RNA/mRNA GPT model | None |
| `mrna_config_path` | Path to model configuration JSON | None |
| `use_pretrained_mrna` | Whether to use pre-trained decoder | False |
| `mrna_lora_r` | LoRA rank for mRNA decoder | 16 |
| `mrna_lora_alpha` | LoRA alpha for mRNA decoder | 32 |
| `fusion_dim` | Dimension of fusion layer | 512 |

## 📊 Data Format

### JSONL Training Data Format

```json
{
  "gcf_id": "GCF_000005825.2",
  "protein_id": "WP_367670421.1",
  "protein_sequence": "MNRISTTITTTITITTGNGAG...",
  "mrna_sequence": "AUGGAACGCAUCUCAACCAUC...",
  "organism_name": "Escherichia coli",
  "order": "Enterobacterales",
  "family": "Enterobacteriaceae", 
  "genus": "Escherichia",
  "species": "coli"
}
```

### Taxonomy CSV Format

```csv
Accession,Organism Name,order,family,genus,species
GCF_000005825.2,Escherichia coli,Enterobacterales,Enterobacteriaceae,Escherichia,coli
```

## 🧠 Model Architecture Details

### Parameter Efficiency with LoRA

The framework uses LoRA (Low-Rank Adaptation) for all large language models:

- **SciBERT**: Only LoRA parameters trainable (~1-2M parameters)
- **ESM-2**: Only LoRA parameters trainable (~1-2M parameters)  
- **Pre-trained mRNA GPT**: Only LoRA parameters trainable (~1-2M parameters)
- **Fusion Module**: Fully trainable (~2M parameters)

**Total trainable parameters**: ~6-8M (vs. ~1B+ for full fine-tuning)

### Pre-trained Decoder Integration

The mRNA decoder can integrate with various pre-trained RNA models:

1. **GenRNA**: General RNA language model
2. **mRNA GPT**: Specialized mRNA language model
3. **Custom RNA Models**: Any GPT-style RNA model

The integration works by:
1. Loading pre-trained weights
2. Adding LoRA adapters to attention and feedforward layers
3. Using a context projector to align fused embeddings
4. Direct feeding of context through transformer layers

## 📈 Model Performance

### Parameter Statistics

| Component | Parameters | Trainable | Memory |
|-----------|------------|-----------|---------|
| SciBERT + LoRA | ~110M | ~1M | ~440MB |
| ESM-2 + LoRA | ~650M | ~1M | ~2.6GB |
| Fusion Module | ~2M | ~2M | ~8MB |
| mRNA GPT + LoRA | ~124M | ~1M | ~500MB |
| **Total** | **~886M** | **~5M** | **~3.5GB** |

### Training Efficiency

- **Memory Usage**: ~4GB GPU memory for batch size 8
- **Training Speed**: ~2-3 hours per epoch on single A100
- **Convergence**: Typically converges within 5-10 epochs

## 🔧 Advanced Usage

### Custom Pre-trained Models

To use your own pre-trained RNA model:

1. **Prepare model files**:
   ```
   your_model.pt          # Model weights
   your_config.json       # Model configuration
   ```

2. **Configuration example**:
   ```json
   {
     "vocab_size": 68,
     "n_positions": 3000,
     "n_embd": 512,
     "n_layer": 6,
     "n_head": 8,
     "n_inner": 2048
   }
   ```

3. **Load in Pro2RNA**:
   ```python
   model = Pro2RNAModel(
       mrna_pretrained_model_path="your_model.pt",
       mrna_config_path="your_config.json",
       use_pretrained_mrna=True
   )
   ```

### LoRA Configuration

Adjust LoRA parameters for different model sizes:

```python
# For smaller models
mrna_lora_r=8
mrna_lora_alpha=16

# For larger models  
mrna_lora_r=32
mrna_lora_alpha=64

# Target specific modules
mrna_target_modules=["c_attn", "c_proj", "c_fc", "mlp"]
```

## 📊 Citation

If you use Pro2RNA in your research, please cite:

```bibtex
@article{pro2rna2024,
  title={Pro2RNA: Multimodal Framework for Organism-Specific mRNA Generation},
  author={Your Name and Others},
  journal={Your Journal},
  year={2024}
}
```
