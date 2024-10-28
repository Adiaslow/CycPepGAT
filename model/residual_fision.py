import torch
import torch.nn as nn

class ResidualFusion(nn.Module):
    def __init__(self, hidden_channels, residual_features, fusion_type='cross_attention'):
        super().__init__()
        self.fusion_type = fusion_type
        self.fusion_layer = nn.Linear(hidden_channels + residual_features, hidden_channels)

    def forward(self, x, residual_features):
        if self.fusion_type == 'cross_attention':
            return self.fusion_layer(torch.cat([x, residual_features], dim=-1))

        elif self.fusion_type == 'dense':
            return self.fusion_layer(torch.cat([x, residual_features], dim=-1))

        else:
            raise ValueError(f"Unsupported fusion type: {self.fusion_type}")
