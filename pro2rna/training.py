from typing import Optional, List
from dataclasses import field, dataclass
import logging
import subprocess
import pathlib
import torch
import shutil
import glob
import os
import logging
import numpy as np

import transformers
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR, set_seed
from transformers import Trainer, AutoTokenizer

import torch.nn as nn
from pro2rna.model.model import RevProtein
from pro2rna.training_data import DataArguments, GCFDataset, custom_collate_fn

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    remove_unused_columns: bool = field(default=False)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={
            "help": "Compress the quantization statistics through double quantization."
        },
    )
    quant_type: str = field(
        default="nf4",
        metadata={
            "help": "Quantization data type to use. Should be one of `fp4` or `nf4`."
        },
    )


@dataclass
class ModelArguments:
    esm_name_or_path: str = field(default="/raid_elmo/home/lr/zym/protein2rna/checkpoints/esm2_t33_650M_UR50D.pt")
    species_model:str = field(default=None)
    decoder_type: str = field(
        default="mlp",
        metadata={
            "help": "type of decoder to predict the codons"
        },
    )
    decoder_path: str = field(
        default=False,
        metadata={
            "help": "path to RNAdecoder checkpoint path"
        },
    )
    embedding_size: int = field(default=768)
    hidden_dim: int = field(default=3072)
    num_heads: int = field(default=12)
    num_decoder_layers: int = field(default=12)
    max_length: int = field(default=512)
    latent_embed_dim: int = field(default=256)
    temp: float = field(default=0.07)
    RNA_config_path: str = field(
        default=None,
        metadata={
            "help": "path to RNAdecoder config file"
        },
    )


class RevProteinTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        确保loss被正确提取，兼容新版本Transformers
        """
        labels = inputs.pop('labels')
        outputs = model(labels=labels, **inputs)
        loss = outputs.get('loss')
        return (loss, outputs) if return_outputs else loss

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.model.save_model_weights(os.path.join(output_dir, "pytorch_model.bin"))




if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    
    
    parser = transformers.HfArgumentParser(
        (TrainingArguments, ModelArguments, DataArguments)
    )

    training_args, model_args, data_args, _ = parser.parse_args_into_dataclasses(
        return_remaining_strings=True
    )
    
    print(f"training_args: {training_args}")
    print(f"model_args: {model_args}")
    print(f"data_args: {data_args}")
    print(f'seed: {training_args.seed}')
    set_seed(training_args.seed)

    train_dataset = GCFDataset(data_args, split="train", max_length=model_args.max_length)
    valid_dataset = GCFDataset(data_args, split="valid", max_length=model_args.max_length)

    model = RevProtein(model_args)
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Amount of Trainable parameters: {num_trainable_params/1e6}M")

    trainer = RevProteinTrainer(
        model=model,                     
        args=training_args,               
        train_dataset=train_dataset,           
        eval_dataset=valid_dataset,      
        data_collator=custom_collate_fn,
    )

    if list(pathlib.Path(training_args.output_dir).glob(f"{PREFIX_CHECKPOINT_DIR}-*")):
        print("resume")
        trainer.train(resume_from_checkpoint=True)
    else:
        print("from scratch")
        trainer.train()

    trainer.save_state()
