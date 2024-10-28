import torch
import torch.nn as nn
from torch.nn import MultiheadAttention
from torch_geometric.nn import global_add_pool

class GlobalAttentionPooling(nn.Module):
    def __init__(self, hidden_channels, num_heads):
        super().__init__()
        self.attention = MultiheadAttention(hidden_channels, num_heads)

    def forward(self, x, batch):
        attention_output = self.attention(x)
        return global_add_pool(attention_output, batch)
