import torch
import torch.nn as nn
from torch_geometric.nn.aggr import AttentionalAggregation
from torch_geometric.nn import GATConv
import torch.nn.functional as F

class ModularGraphAttentionTransformer(nn.Module):
    def __init__(self, node_features, edge_features, hidden_channels, **kwargs):
        super().__init__()
        # Store config
        self.config = {k: v for k, v in locals().items() if k != 'self'}
        self.config.update(kwargs)

        # Initialize components
        self.hidden_channels = hidden_channels
        self.edge_features = edge_features
        self.use_edge_features = kwargs.get('use_edge_features', False) and edge_features > 0
        self.num_heads = kwargs.get('num_heads', 8)

        # Initial node embedding
        self.node_embedding = nn.Sequential(
            nn.Linear(node_features, hidden_channels),
            nn.GELU(),
            nn.BatchNorm1d(hidden_channels)
        )

        # Edge embedding if using edge features
        if self.use_edge_features:
            self.edge_embedding = nn.Sequential(
                nn.Linear(edge_features, hidden_channels),
                nn.GELU(),
                nn.BatchNorm1d(hidden_channels)
            )

        # Main GAT layers
        self.layers = nn.ModuleList()
        for _ in range(kwargs.get('num_layers', 4)):
            layer = GATConv(
                in_channels=hidden_channels,
                out_channels=hidden_channels // self.num_heads,
                heads=self.num_heads,
                dropout=kwargs.get('dropout', 0.1),
                edge_dim=hidden_channels if self.use_edge_features else None,
                concat=True,
                add_self_loops=True
            )
            self.layers.append(layer)

        # Layer normalization
        self.layer_norms = nn.ModuleList([
            nn.BatchNorm1d(hidden_channels) for _ in range(kwargs.get('num_layers', 4))
        ])

        # Attention pooling
        self.pool = AttentionalAggregation(
            gate_nn=nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                nn.GELU(),
                nn.Linear(hidden_channels, 1)
            ),
            nn=None  # No transform needed as we already have hidden_channels dimension
        )

        # Output layers
        self.output_layers = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels * 2),
            nn.GELU(),
            nn.Dropout(kwargs.get('dropout', 0.1)),
            nn.Linear(hidden_channels * 2, 1)
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Initial node embedding
        x = self.node_embedding(x)

        # Process edge features if present
        edge_attr = None
        if self.use_edge_features and hasattr(data, 'edge_attr'):
            edge_attr = self.edge_embedding(data.edge_attr)

        # Main layers
        for layer, norm in zip(self.layers, self.layer_norms):
            # Apply GAT layer
            x_res = x  # Store residual
            x = layer(x, edge_index, edge_attr=edge_attr)
            x = norm(x)
            x = F.gelu(x)
            x = x + x_res  # Add residual

        # Global pooling
        x = self.pool(x, batch)

        # Output projection
        return self.output_layers(x)
