import torch
import torch.nn as nn
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
from torch_geometric.nn import Set2Set, TopKPooling
from torch_geometric.nn import AddLaplacianEigenvectorPE, AddRandomWalkPE
from torch_geometric.data import Data


class PositionalEncoding(nn.Module):
    """Flexible positional encoding module using PyTorch Geometric's implementations."""
    def __init__(
        self,
        hidden_channels: int,
        pe_type: str = 'laplacian',  # ['laplacian', 'random_walk', 'none']
        num_pe_features: int = 16,
        walk_length: int = 8,
        is_undirected: bool = False
    ):
        super().__init__()
        self.pe_type = pe_type
        self.hidden_channels = hidden_channels

        if pe_type == 'laplacian':
            self.pe_encoder = AddLaplacianEigenvectorPE(
                k=num_pe_features,
                attr_name='pe',  # Store PE in separate attribute
                is_undirected=is_undirected
            )
            self.pe_dim = num_pe_features
        elif pe_type == 'random_walk':
            self.pe_encoder = AddRandomWalkPE(
                walk_length=walk_length,
                attr_name='pe'  # Store PE in separate attribute
            )
            self.pe_dim = walk_length
        else:
            self.pe_encoder = None
            self.pe_dim = 0

        if pe_type != 'none':
            # Project PE to hidden dimension
            self.pe_projection = nn.Linear(self.pe_dim, hidden_channels)
            self.layer_norm = nn.LayerNorm(hidden_channels)

    def forward(self, x, edge_index, batch):
        if self.pe_type == 'none':
            return x

        # Create temporary Data object for PE computation
        temp_data = Data(
            x=x,
            edge_index=edge_index,
            batch=batch,
            num_nodes=x.size(0)
        )

        # Compute PE
        temp_data = self.pe_encoder(temp_data)

        # Get PE features and project them
        pe = self.pe_projection(temp_data.pe)
        pe = self.layer_norm(pe)

        # Add PE to input features
        x = x + pe

        return x
