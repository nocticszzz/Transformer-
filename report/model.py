import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ScaledDotProductAttention(nn.Module):
    """
    缩放点积注意力（Scaled Dot-Product Attention）
    公式: Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V
    """
    def __init__(self, d_k):
        super().__init__()
        self.d_k = d_k

    def forward(self, Q, K, V, mask=None):
        # Q, K, V shape: [batch_size, n_heads, len_q/k/v, d_k]
        
        # 1. Matmul Q and K^T
        scores = torch.matmul(Q, K.transpose(-2, -1)) 
        
        # 2. Scale
        scores = scores / math.sqrt(self.d_k)
        
        # 3. Mask (Optional, usually for padding)
        if mask is not None:

            scores = scores.masked_fill(mask == 0, -1e9)
        
        # 4. Softmax
        attn_weights = F.softmax(scores, dim=-1)
        
        # 5. Matmul with V
        output = torch.matmul(attn_weights, V)
        
        return output, attn_weights

class MultiHeadAttention(nn.Module):
    """
    多头注意力（Multi-Head Attention）
    """
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # W_Q, W_K, W_V Linear layers
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        
        self.fc_out = nn.Linear(d_model, d_model)
        self.attention = ScaledDotProductAttention(self.d_k)
        
    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        
        # 1. Linear projections and split heads
        Q = self.w_q(q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(k).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(v).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        out, attn_weights = self.attention(Q, K, V, mask)
        
        # 2. 合并多头
        out = out.transpose(1, 2).contiguous()
        
        out = out.view(batch_size, -1, self.d_model) 
        
        return self.fc_out(out), attn_weights

class PositionwiseFeedForward(nn.Module):
    """
    前馈网络（Position-wise FFN）
    FFN(x) = max(0, xW1 + b1)W2 + b2
    """
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU() # or nn.ReLU()

    def forward(self, x):
        return self.fc2(self.dropout(self.activation(self.fc1(x))))

class EncoderLayer(nn.Module):
    """
    编码器层
    包含: Self-Attention -> Add&Norm -> FFN -> Add&Norm
    """
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        
        # LayerNorm and Dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # 1. Sublayer 1: Self Attention
        attn_output, _ = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout1(attn_output)) # Add & Norm
        
        # 2. Sublayer 2: FFN
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_output)) # Add & Norm
        
        return x

class PositionalEncoding(nn.Module):
    """
    位置编码（正弦/学习式）
    """
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register as buffer 
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        return x + self.pe[:, :x.size(1)]
    
# 整体模型
class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, d_ff, n_layers, n_classes, dropout=0.1, max_len=500):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.layers = nn.ModuleList([EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(d_model, n_classes)

    def forward(self, x, mask=None):
        batch_size = x.size(0)
        x = self.embedding(x)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.pos_encoder(x)
        x = self.dropout(x)
        
        if mask is not None:
            cls_mask = torch.ones((batch_size, 1), device=x.device)
            mask = torch.cat((cls_mask, mask), dim=1)
            mask = mask.unsqueeze(1).unsqueeze(2)
        
        for layer in self.layers:
            x = layer(x, mask)
            
        cls_output = x[:, 0, :] 
        return self.fc_out(cls_output)