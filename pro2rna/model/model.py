import torch
import torch.nn as nn
from torch.nn import TransformerDecoder, TransformerDecoderLayer
import torch.nn.functional as F
import esm
from pro2rna.model.projectors import build_patch_mlp_projector
from pro2rna.model.GPT_model import GPTConfig, GPT
from pro2rna.model.attention import TransformerEncoder
from pro2rna.utils.tokenizer import get_tokenizer, get_base_tokenizer
from pro2rna.utils.utils import sample_codon_id_seqs
from torch.nn.functional import softmax
import gc
from transformers import AutoTokenizer, AutoModel
from pro2rna.calm.alphabet import Alphabet
import json
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
import math

class RevProtein(nn.Module):
    def __init__(self, config):
        super(RevProtein, self).__init__()
        self.decoder_type = config.decoder_type
        self.decoder_path = config.decoder_path

        # 添加LoRA配置参数
        self.esm_lora_config = getattr(config, 'esm_lora_config', {
            'r': 16, 'alpha': 32, 'dropout': 0.1, 'target_modules': ['query', 'value', 'key', 'dense']
        })
        self.species_lora_config = getattr(config, 'species_lora_config', {
            'r': 16, 'alpha': 32, 'dropout': 0.1, 'target_modules': ['query', 'value', 'key', 'dense']
        })
        self.decoder_lora_config = getattr(config, 'decoder_lora_config', {
            'r': 16, 'alpha': 32, 'dropout': 0.1, 'target_modules': ['c_attn', 'c_proj']
        })

        # 加载ESM模型
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.esm_model, self.esm_alphabet = self._init_esm_weights(config.esm_name_or_path)
        self.codon_alphabet = Alphabet.from_architecture("CodonModel")
        self.padding_idx = self.codon_alphabet.padding_idx

        self.species_model = None
        if config.species_model == 'scibert':
            self.species_tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_cased')
            self.species_model = AutoModel.from_pretrained('allenai/scibert_scivocab_cased')
        elif config.species_model == 'biobert':
            self.species_tokenizer = AutoTokenizer.from_pretrained('dmis-lab/biobert-base-cased-v1.2')
            self.species_model = AutoModel.from_pretrained('dmis-lab/biobert-base-cased-v1.2')
        
        if self.species_model:
            self._apply_species_lora()
        
        num_trainable_params = sum(p.numel() for p in self.species_model.parameters()) if self.species_model else 0

        self.max_length = config.max_length
        
        self._apply_esm_lora()
        
        self.batch_converter = self.esm_alphabet.get_batch_converter()
        self.esm_model.eval()

        self.num_projector_layers = 2
        self.prot_embed_dim = self.esm_model.embed_dim   # hidden size of esm
        self.embedding_size = config.embedding_size # hidden size of bert models
        self.hidden_dim = config.hidden_dim
        self.latent_embed_dim = config.latent_embed_dim
        self.num_heads = config.num_heads
        self.num_decoder_layers = config.num_decoder_layers
        self.esm_model_layers = len(self.esm_model.layers)

        self.projector = build_patch_mlp_projector(
            input_hidden_size=self.prot_embed_dim,
            lm_hidden_size=self.embedding_size,
            num_layers=self.num_projector_layers,
        )

        # concat feature
        self.prot_species_to_rna = build_patch_mlp_projector(
            input_hidden_size=self.prot_embed_dim + self.embedding_size,
            lm_hidden_size=self.embedding_size,
            num_layers=self.num_projector_layers,
        )

        self.prot_species_projector = build_patch_mlp_projector(
            input_hidden_size=self.embedding_size,
            lm_hidden_size=self.embedding_size,
            num_layers=self.num_projector_layers,
        )


        self.vocab_size = len(self.codon_alphabet.tok_to_idx)

        self.protein_to_rna = nn.Linear(self.embedding_size, self.embedding_size)
        self.rna_to_protein = nn.Linear(self.embedding_size, self.embedding_size)   
        self.output_layer = nn.Linear(self.embedding_size, self.vocab_size)


        self.temp = nn.Parameter(torch.ones([]) * config.temp) 
        self.protein_proj = nn.Linear(self.embedding_size, self.latent_embed_dim)
        self.codon_proj = nn.Linear(self.embedding_size, self.latent_embed_dim) 
        self.loss_fcn = nn.CrossEntropyLoss(ignore_index=self.padding_idx)

        self._keys_to_ignore_on_save = []

        # load GenerRNA model
        if self.decoder_type == 'RNAdecoder':
            with open(config.RNA_config_path, 'r') as f:
                RNA_config = json.load(f)
            f.close()
            
            self.RNA_config = RNA_config
            self.RNA_embedding_dim = self.RNA_config['n_embd']
            self.RNAdecoder = self.init_RNAdecoder(config, self.RNA_config)
            self._apply_decoder_lora()
            
            self.RNA_projector = build_patch_mlp_projector(
                input_hidden_size=self.prot_embed_dim + self.embedding_size,
                lm_hidden_size=self.RNA_embedding_dim,
                num_layers=self.num_projector_layers,
            )
            self.lm_head = nn.Linear(self.RNA_embedding_dim, self.vocab_size)

    def _apply_esm_lora(self):
        from peft.tuners.lora import LoraLayer, Linear
        import torch.nn as nn
        
        # 获取LoRA配置
        r = self.esm_lora_config['r']
        alpha = self.esm_lora_config['alpha']
        dropout = self.esm_lora_config['dropout']
        target_modules = self.esm_lora_config['target_modules']
        
        # 手动为ESM模型的注意力层添加LoRA
        for layer_idx, layer in enumerate(self.esm_model.layers):
            # 为self-attention的query, key, value添加LoRA
            if 'query' in target_modules or 'q_proj' in target_modules:
                self._add_lora_to_linear(layer.self_attn.q_proj, r, alpha, dropout)
            if 'key' in target_modules or 'k_proj' in target_modules:
                self._add_lora_to_linear(layer.self_attn.k_proj, r, alpha, dropout)
            if 'value' in target_modules or 'v_proj' in target_modules:
                self._add_lora_to_linear(layer.self_attn.v_proj, r, alpha, dropout)
            if 'dense' in target_modules or 'out_proj' in target_modules:
                self._add_lora_to_linear(layer.self_attn.out_proj, r, alpha, dropout)
        
        # 冻结基础模型参数
        self._freeze_esm_base_params()
        
        print(f"Applied LoRA to ESM model with r={r}, alpha={alpha}")
    
    def _add_lora_to_linear(self, linear_layer, r, alpha, dropout):
        """为线性层添加LoRA适配器"""
        import torch.nn as nn
        
        # 创建LoRA的A和B矩阵
        in_features = linear_layer.in_features
        out_features = linear_layer.out_features
        
        # LoRA的低秩分解：ΔW = BA，其中A是r×in_features，B是out_features×r
        lora_A = nn.Parameter(torch.randn(r, in_features) / math.sqrt(r))
        lora_B = nn.Parameter(torch.zeros(out_features, r))
        lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # 将LoRA参数添加到线性层
        linear_layer.lora_A = lora_A
        linear_layer.lora_B = lora_B  
        linear_layer.lora_dropout = lora_dropout
        linear_layer.lora_scaling = alpha / r
        
        # 修改线性层的forward方法
        original_forward = linear_layer.forward
        
        def lora_forward(x):
            result = original_forward(x)
            # x的形状: (..., in_features)
            # lora_A的形状: (r, in_features)
            # lora_B的形状: (out_features, r)
            lora_out = lora_dropout(x) @ lora_A.t()  # (..., r)
            lora_out = lora_out @ lora_B.t()  # (..., out_features)
            lora_result = lora_out * linear_layer.lora_scaling
            return result + lora_result
        
        linear_layer.forward = lora_forward

    def _apply_species_lora(self):
        """为species模型应用LoRA微调"""
        # 创建LoRA配置
        lora_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=self.species_lora_config['r'],
            lora_alpha=self.species_lora_config['alpha'],
            lora_dropout=self.species_lora_config['dropout'],
            target_modules=self.species_lora_config['target_modules'],
            bias="none"
        )
        
        # 应用LoRA
        self.species_model = get_peft_model(self.species_model, lora_config)
        
        # 冻结基础模型参数，只保留LoRA参数可训练
        self._freeze_species_base_params()
        
        print(f"Applied LoRA to species model with r={self.species_lora_config['r']}, alpha={self.species_lora_config['alpha']}")

    def _apply_decoder_lora(self):
        # 创建LoRA配置
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.decoder_lora_config['r'],
            lora_alpha=self.decoder_lora_config['alpha'],
            lora_dropout=self.decoder_lora_config['dropout'],
            target_modules=self.decoder_lora_config['target_modules'],
            bias="none"
        )
        
        # 应用LoRA
        self.RNAdecoder = get_peft_model(self.RNAdecoder, lora_config)
        
        # 冻结基础模型参数，只保留LoRA参数可训练
        self._freeze_decoder_base_params()
        
        print(f"Applied LoRA to RNAdecoder with r={self.decoder_lora_config['r']}, alpha={self.decoder_lora_config['alpha']}")

    def _freeze_esm_base_params(self):
        """冻结ESM基础模型参数，只保留LoRA参数可训练"""
        for name, param in self.esm_model.named_parameters():
            if "lora_" not in name:
                param.requires_grad = False
            else:
                param.requires_grad = True

    def _freeze_species_base_params(self):
        """冻结species基础模型参数，只保留LoRA参数可训练"""
        for name, param in self.species_model.named_parameters():
            if "lora" not in name.lower():
                param.requires_grad = False
            else:
                param.requires_grad = True

    def _freeze_decoder_base_params(self):
        """冻结decoder基础模型参数，只保留LoRA参数可训练"""
        for name, param in self.RNAdecoder.named_parameters():
            if "lora" not in name.lower():
                param.requires_grad = False
            else:
                param.requires_grad = True

    def unfreeze_lora_params(self):
        """确保所有LoRA参数可训练"""
        # ESM模型LoRA参数
        for name, param in self.esm_model.named_parameters():
            if "lora_" in name:
                param.requires_grad = True
        
        # Species模型LoRA参数
        if self.species_model:
            for name, param in self.species_model.named_parameters():
                if "lora" in name.lower():
                    param.requires_grad = True
        
        # RNAdecoder LoRA参数
        if hasattr(self, 'RNAdecoder'):
            for name, param in self.RNAdecoder.named_parameters():
                if "lora" in name.lower():
                    param.requires_grad = True

    def get_trainable_parameters(self):
        """获取可训练参数的详细信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        esm_lora_params = 0
        species_lora_params = 0
        decoder_lora_params = 0
        
        # 统计ESM LoRA参数
        for name, param in self.esm_model.named_parameters():
            if param.requires_grad and "lora_" in name:
                esm_lora_params += param.numel()
        
        # 统计species LoRA参数
        if self.species_model:
            for name, param in self.species_model.named_parameters():
                if param.requires_grad and "lora" in name.lower():
                    species_lora_params += param.numel()
        
        # 统计decoder LoRA参数
        if hasattr(self, 'RNAdecoder'):
            for name, param in self.RNAdecoder.named_parameters():
                if param.requires_grad and "lora" in name.lower():
                    decoder_lora_params += param.numel()
        
        return {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'trainable_percentage': (trainable_params / total_params) * 100,
            'esm_lora_params': esm_lora_params,
            'species_lora_params': species_lora_params,
            'decoder_lora_params': decoder_lora_params
        }

    def save_lora_weights(self, save_path):
        """保存LoRA权重"""
        import os
        os.makedirs(save_path, exist_ok=True)
        
        # 保存ESM LoRA权重
        esm_lora_weights = {}
        for name, param in self.esm_model.named_parameters():
            if "lora_" in name:
                esm_lora_weights[name] = param.data
        
        if esm_lora_weights:
            esm_lora_path = os.path.join(save_path, 'esm_lora_weights.pt')
            torch.save(esm_lora_weights, esm_lora_path)
        
        # 保存species LoRA权重
        if self.species_model and hasattr(self.species_model, 'save_pretrained'):
            species_lora_path = os.path.join(save_path, 'species_lora')
            self.species_model.save_pretrained(species_lora_path)
        
        # 保存decoder LoRA权重
        if hasattr(self, 'RNAdecoder') and hasattr(self.RNAdecoder, 'save_pretrained'):
            decoder_lora_path = os.path.join(save_path, 'decoder_lora')
            self.RNAdecoder.save_pretrained(decoder_lora_path)
        
        print(f"Saved LoRA weights to {save_path}")

    def load_lora_weights(self, load_path):
        """加载LoRA权重"""
        import os
        from peft import PeftModel
        
        # 加载ESM LoRA权重
        esm_lora_path = os.path.join(load_path, 'esm_lora_weights.pt')
        if os.path.exists(esm_lora_path):
            esm_lora_weights = torch.load(esm_lora_path, map_location=self.device)
            # 加载权重到模型中
            for name, param in self.esm_model.named_parameters():
                if name in esm_lora_weights:
                    param.data = esm_lora_weights[name]
        
        # 加载species LoRA权重
        species_lora_path = os.path.join(load_path, 'species_lora')
        if self.species_model and os.path.exists(species_lora_path):
            self.species_model = PeftModel.from_pretrained(self.species_model, species_lora_path)
        
        # 加载decoder LoRA权重
        decoder_lora_path = os.path.join(load_path, 'decoder_lora')
        if hasattr(self, 'RNAdecoder') and os.path.exists(decoder_lora_path):
            self.RNAdecoder = PeftModel.from_pretrained(self.RNAdecoder, decoder_lora_path)
        
        print(f"Loaded LoRA weights from {load_path}")

    # MLP decoder
    def forward(self, labels, **kwargs):
        torch.cuda.empty_cache()
        batch_protein_sequence = kwargs['protein_sequence']
        prompts = kwargs['prompts']
        codon_labels = labels.to(self.device)
        
        # 1) extrcat protein embedding(per residue) and whole protein sequence embedding
        batch_protein_tokens, protein_embeddings = self.encode_protein_sequence(batch_protein_sequence)
        # protein_embeddings = self.projector(protein_embeddings)

        # 2) fuse protein embedding and species embedding
        if self.species_model != None:
            species_embeddings = self.extract_species_embedding(prompts)
            # protein_sequence_representations = protein_sequence_representations + species_embeddings

            # species_embeddings = species_embeddings.unsqueeze(1)    # (B, 1, D)
            # protein_embeddings = protein_embeddings + species_embeddings

            # concat feature
            species_embeddings = species_embeddings.unsqueeze(1).expand(-1, protein_embeddings.size(1), -1)   # (B, L, D)
            combined_emb = torch.cat((protein_embeddings, species_embeddings), dim=2)
            if self.decoder_type == "RNAdecoder":
                protein_embeddings = self.RNA_projector(combined_emb)
            else: 
                protein_embeddings = self.prot_species_to_rna(combined_emb) # (B, L, D)
            # (B, L, D) -> (B, L+1, D)
            
            # species_embeddings = species_embeddings.unsqueeze(1)
            # protein_embeddings = self.projector(protein_embeddings)
            # combined_emb = torch.cat((species_embeddings, protein_embeddings), dim=1)
            # protein_embeddings = self.prot_species_projector(combined_emb) # (B, L+1, D)
            # print(f"protein_embeddings: {protein_embeddings.shape}")
        


        # 3) output the logtis
        if self.decoder_type == 'mlp':
            logits = self.output_layer(protein_embeddings)
        elif self.decoder_type == 'RNAdecoder':
            for block in self.RNAdecoder.transformer.h:
                protein_embeddings = block(protein_embeddings)
            protein_embeddings = self.RNAdecoder.transformer.ln_f(protein_embeddings)
            logits = self.lm_head(protein_embeddings)
        
        # post-refine the prediction
        total_loss = 0
        
        # 确保codon_labels和logits的形状匹配
        # logits: (batch_size, seq_len, vocab_size)
        # codon_labels: (batch_size, seq_len)
        if len(codon_labels.shape) == 3 and codon_labels.shape[1] == 1:
            codon_labels = codon_labels.squeeze(1)
        
        batch_size, seq_len, vocab_size = logits.shape
        logits_flat = logits.contiguous().view(-1, vocab_size)  # (batch_size * seq_len, vocab_size)
        labels_flat = codon_labels.contiguous().view(-1)  # (batch_size * seq_len,)
        
        total_loss += self.loss_fcn(logits_flat, labels_flat)

        return {
            'logits': logits,
            'loss': total_loss,
        }


    @torch.no_grad()
    def encode_protein_sequence(self, batch_protein_sequence):
        self.esm_model.eval()
        protein_data = [("", seq) for seq in batch_protein_sequence]
        batch_labels, batch_strs, batch_tokens  = self.batch_converter(protein_data)

        batch_tokens = batch_tokens.to(self.device)
        batch_lens = (batch_tokens != self.esm_alphabet.padding_idx).sum(1) # batch_lens = seq_length + 2 (<cls> and <eos>)
        # padded_batch_tokens = torch.full((batch_tokens.size(0), self.max_length), self.esm_alphabet.padding_idx, dtype=torch.long, device=self.device)
        # print(f"padded_batch_tokens shape: {padded_batch_tokens.shape}")
        # 将原始tokens复制到新的padded_batch_tokens中
        # for i, seq_len in enumerate(batch_lens):
        #     print(f'seq_len: {seq_len}')
        #     padded_batch_tokens[i, :seq_len-1] = batch_tokens[i, :seq_len-1]

        protein_repr = self.esm_model(batch_tokens, repr_layers=[self.esm_model_layers], return_contacts=True)
        
        protein_embeddings = protein_repr["representations"][self.esm_model_layers]
        return batch_tokens, protein_embeddings
    
    @torch.no_grad()
    def extract_species_embedding(self, prompts):
        inputs = self.species_tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = inputs.to(self.device)
        with torch.no_grad():
            outputs = self.species_model(**inputs)

        # del inputs
        return outputs.last_hidden_state[:,0,:].detach()
    
    def calculate_contrastive_loss(self, predicted_features, ground_truth_features):
        loss_function = nn.CosineEmbeddingLoss()
        labels = torch.ones(predicted_features.size(0)).to(predicted_features.device)
        loss = loss_function(predicted_features, ground_truth_features, labels)     # maximize the cosine similarity
        return loss
    
    def _init_esm_weights(self, esm_name_or_path):
        return esm.pretrained.load_model_and_alphabet(esm_name_or_path)
    
    def save_model_weights(self, save_path):
        """
        保存权重，不包括esm、species_model和codon_model的基础权重，但包含LoRA权重
        """
        model_dict = {}
        
        # 保存所有非预训练模型的权重
        for k, v in self.state_dict().items():
            if not any(prefix in k for prefix in ['esm_model.base_model', 'species_model.base_model', 'codon_model']):
                model_dict[k] = v
        
        torch.save(model_dict, save_path)
        print(f"模型权重（包含LoRA）已保存到: {save_path}")

    def load_model_weights(self, model_path: str, strict: bool = False):
        """ 
        加载模型权重，兼容LoRA权重
        """
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # 过滤掉不存在的键
        model_dict = {}
        for k, v in checkpoint.items():
            if k in self.state_dict():
                model_dict[k] = v
        
        self.load_state_dict(model_dict, strict=strict)
        print(f"模型权重已从 {model_path} 加载")
    
    def freeze_all_except_lora(self):
        """冻结所有参数，除了LoRA参数"""
        for name, param in self.named_parameters():
            if "lora" not in name.lower():
                param.requires_grad = False
            else:
                param.requires_grad = True
        print("已冻结所有参数，除了LoRA参数")
    
    def print_lora_info(self):
        """打印LoRA相关信息"""
        print("\n=== LoRA配置信息 ===")
        
        if hasattr(self, 'esm_lora_config'):
            print(f"ESM LoRA - r: {self.esm_lora_config['r']}, alpha: {self.esm_lora_config['alpha']}")
            print(f"ESM目标模块: {self.esm_lora_config['target_modules']}")
        
        if hasattr(self, 'species_lora_config') and self.species_model:
            print(f"Species LoRA - r: {self.species_lora_config['r']}, alpha: {self.species_lora_config['alpha']}")
            print(f"Species目标模块: {self.species_lora_config['target_modules']}")
        
        if hasattr(self, 'decoder_lora_config') and hasattr(self, 'RNAdecoder'):
            print(f"Decoder LoRA - r: {self.decoder_lora_config['r']}, alpha: {self.decoder_lora_config['alpha']}")
            print(f"Decoder目标模块: {self.decoder_lora_config['target_modules']}")
        
        # 显示参数统计
        param_info = self.get_trainable_parameters()
        print(f"\n参数统计:")
        print(f"  总参数: {param_info['total_params']:,}")
        print(f"  可训练参数: {param_info['trainable_params']:,} ({param_info['trainable_percentage']:.2f}%)")
        if param_info['esm_lora_params'] > 0:
            print(f"  ESM LoRA参数: {param_info['esm_lora_params']:,}")
        if param_info['species_lora_params'] > 0:
            print(f"  Species LoRA参数: {param_info['species_lora_params']:,}")
        if param_info['decoder_lora_params'] > 0:
            print(f"  Decoder LoRA参数: {param_info['decoder_lora_params']:,}")

    def init_RNAdecoder(self, config, RNA_config):
        checkpoint = torch.load(config.decoder_path, map_location=self.device)
        gptconf = GPTConfig(**checkpoint['model_args'])
        model = GPT(gptconf)
        state_dict = checkpoint['model']
        unwanted_prefix = '_orig_mod.'
        for k,v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
        
        return model
    
    # def init_generRNA(self, config, GenerRNA_config):
    #     ckpt_path = GenerRNA_config['ckpt_path']
    #     checkpoint = torch.load(ckpt_path, map_location=self.device)
    #     gptconf = GPTConfig(**checkpoint['model_args'])
    #     model = GPT(gptconf)
    #     state_dict = checkpoint['model']
    #     unwanted_prefix = '_orig_mod.'
    #     for k,v in list(state_dict.items()):
    #         if k.startswith(unwanted_prefix):
    #             state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    #     model.load_state_dict(state_dict)

    #     # freeze all params
    #     for name, param in model.transformer.h[i].named_parameters():
    #         param.requires_grad = True
    #     return model



if __name__ == "__main__":    
    protein_to_codon_transformer = RevProtein(vocab_size=64, embedding_size=768, hidden_dim=3072, nhead=12, num_decoder_layers=12)
    protein_sequence = "METHIONINE"  # 示例蛋白质序列
    codon_sequence = protein_to_codon_transformer.generate_codon_sequence(protein_sequence)
    print("Generated Codon Sequence:", codon_sequence)
