"""
Pro2RNA: A multimodal framework for organism-specific mRNA generation
Combines SciBERT (taxonomy), ESM-2 (protein), and mRNA-GPT (decoder)

Architecture:
- TaxonomyEncoder: SciBERT + LoRA for taxonomy context encoding
- ProteinEncoder: ESM-2 + LoRA for protein sequence encoding  
- CrossAttentionFusion: Multimodal fusion via cross-attention
- mRNADecoder: mRNA-GPT for autoregressive sequence generation
"""

__version__ = "1.0.0"
__author__ = "Pro2RNA Team"

from .models import (
    Pro2RNAModel,
    TaxonomyEncoder,
    ProteinEncoder,
    CrossAttentionFusion,
    mRNADecoder
)

from .data import (
    Pro2RNADataset,
    TaxonomyPromptGenerator,
    mRNATokenizer,
    collate_fn
)

from .training import (
    Pro2RNATrainer,
    Pro2RNALoss
)

__all__ = [
    # Main model
    "Pro2RNAModel",
    
    # Model components
    "TaxonomyEncoder",
    "ProteinEncoder", 
    "CrossAttentionFusion",
    "mRNADecoder",
    
    # Data processing
    "Pro2RNADataset",
    "TaxonomyPromptGenerator",
    "mRNATokenizer",
    "collate_fn",
    
    # Training
    "Pro2RNATrainer",
    "Pro2RNALoss"
]
