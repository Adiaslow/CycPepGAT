import torch
import torch.nn as nn

class HybridPooling(nn.Module):
    def __init__(self, attention_pool, topk_pool, set2set_pool):
        super().__init__()
        self.attention_pool = attention_pool
        self.topk_pool = topk_pool
        self.set2set_pool = set2set_pool

    def forward(self, x, edge_index, batch):
        # Combine different pooling methods
        x1 = self.attention_pool(x, batch)
        x2 = self.topk_pool(x, edge_index, batch)[0]
        x3 = self.set2set_pool(x, batch)
        return torch.cat([x1, x2, x3], dim=-1)
