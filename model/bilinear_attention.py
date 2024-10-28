import torch
import torch.nn as nn
from torch.nn import Dropout
import torch.nn.functional as F

class BilinearAttention(nn.Module):
    def __init__(self, hidden_channels, dropout=0.1):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(hidden_channels, hidden_channels))
        self.dropout = Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        attention = torch.matmul(torch.matmul(x, self.weight), x.transpose(-2, -1))
        attention = F.softmax(attention, dim=-1)
        attention = self.dropout(attention)
        return torch.matmul(attention, x)
