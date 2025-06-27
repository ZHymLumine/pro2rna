import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadCrossAttention(nn.Module):
    def __init__(self, num_heads, d_model, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)
        
    def forward(self, q, k, v, mask=None):
        bs = q.size(0)
        
        # perform linear operation and split into num_heads
        q = self.q_linear(q).view(bs, -1, self.num_heads, self.d_k).transpose(1,2)
        k = self.k_linear(k).view(bs, -1, self.num_heads, self.d_k).transpose(1,2)
        v = self.v_linear(v).view(bs, -1, self.num_heads, self.d_k).transpose(1,2)
        
        # calculate attention using function we will define next
        scores = torch.matmul(q, k.transpose(-2, -1)) /  math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # apply attention to values
        x = torch.matmul(attn, v).transpose(1, 2).contiguous().view(bs, -1, self.d_model)
        x = self.out(x)
        
        return x


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.cross_attention = MultiHeadCrossAttention(num_heads, d_model, dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.attn_layer_norm = nn.LayerNorm(d_model)
        self.ff_layer_norm = nn.LayerNorm(d_model)
        self.attn_dropout = nn.Dropout(dropout)
        self.ff_dropout = nn.Dropout(dropout)
    
    def forward(self, src, cross, src_mask=None, cross_mask=None):
        # Cross-attention block
        _src = src
        src = self.attn_layer_norm(src)
        attn_output = self.cross_attention(src, cross, cross, mask=src_mask)
        src = _src + self.attn_dropout(attn_output)
        
        # Feed forward block
        _src = src
        src = self.ff_layer_norm(src)
        ff_output = self.feed_forward(src)
        src = _src + self.ff_dropout(ff_output)
        
        return src

class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, dim_feedforward=3072, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, src, cross, src_mask=None, cross_mask=None):
        for layer in self.layers:
            src = layer(src, cross, src_mask, cross_mask)
            
        return self.norm(src)
    
