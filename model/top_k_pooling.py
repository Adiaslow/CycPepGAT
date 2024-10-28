import torch.nn as nn

class TopKPooling(nn.Module):
    def __init__(self, hidden_channels, ratio=0.8):
        super().__init__()
        self.ratio = ratio
        self.topk = TopKPooling(hidden_channels, ratio=ratio)

    def forward(self, x, edge_index, batch):
        pooled_output = self.topk(x, edge_index, batch)
        return pooled_output[0]
