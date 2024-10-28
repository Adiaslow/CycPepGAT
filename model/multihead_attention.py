import math
import torch
import torch.nn as nn
from torch.nn import Linear, Dropout
import torch.nn.functional as F

class MultiheadAttention(nn.Module):
    def __init__(self, hidden_channels, num_heads, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_channels = hidden_channels
        self.head_dim = hidden_channels // num_heads

        self.q_proj = Linear(hidden_channels, hidden_channels)
        self.k_proj = Linear(hidden_channels, hidden_channels)
        self.v_proj = Linear(hidden_channels, hidden_channels)
        self.o_proj = Linear(hidden_channels, hidden_channels)
        self.dropout = Dropout(dropout)

    def forward(self, x):
        batch_size = x.size(0)
        q = self.q_proj(x).view(batch_size, -1, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, -1, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, -1, self.num_heads, self.head_dim)

        attention = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention = F.softmax(attention, dim=-1)
        attention = self.dropout(attention)

        out = torch.matmul(attention, v)
        out = out.view(batch_size, -1, self.hidden_channels)
        out = self.o_proj(out)
        return out
