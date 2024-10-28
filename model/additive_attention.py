import torch
import torch.nn as nn
from torch.nn import Linear, Dropout
import torch.nn.functional as F

class AdditiveAttention(nn.Module):
    def __init__(self, hidden_channels, dropout=0.1):
        super().__init__()
        self.query_proj = Linear(hidden_channels, hidden_channels)
        self.key_proj = Linear(hidden_channels, hidden_channels)
        self.value_proj = Linear(hidden_channels, 1)
        self.dropout = Dropout(dropout)

    def forward(self, x):
        q = self.query_proj(x)
        k = self.key_proj(x)
        scores = self.value_proj(torch.tanh(q + k))
        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        return torch.matmul(attention.transpose(-2, -1), x)
