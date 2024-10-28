import math
import torch
import torch.nn as nn
from torch.nn import Dropout
import torch.nn.functional as F

class DotProductAttention(nn.Module):
    def __init__(self, hidden_channels, dropout=0.1):
        super().__init__()
        self.dropout = Dropout(dropout)
        self.scale = math.sqrt(hidden_channels)

    def forward(self, x):
        attention = torch.matmul(x, x.transpose(-2, -1)) / self.scale
        attention = F.softmax(attention, dim=-1)
        attention = self.dropout(attention)
        return torch.matmul(attention, x)
