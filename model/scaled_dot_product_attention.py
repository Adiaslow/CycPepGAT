import math
import torch
import torch.nn as nn
from torch.nn import Dropout
import torch.nn.functional as F

class ScaledDotProductAttention(nn.Module):
    def __init__(self, hidden_channels, dropout=0.1):
        super().__init__()
        self.dropout = Dropout(dropout)
        self.scale = math.sqrt(hidden_channels)

    def forward(self, q, k, v, mask=None):
        attention = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        if mask is not None:
            attention = attention.masked_fill(mask == 0, -1e9)
        attention = F.softmax(attention, dim=-1)
        attention = self.dropout(attention)
        return torch.matmul(attention, v)
