# @title Peptide Dataset

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
from torch_geometric.data import Dataset
from tqdm import tqdm

from peptide import Peptide

class PeptideDataset(Dataset):
    def __init__(self, csv_file, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(None, transform, pre_transform, pre_filter)
        self.data = pd.read_csv(csv_file)
        self.processed_data = []
        self._indices = None
        self.y_scaler = StandardScaler()
        self.node_feature_scaler = StandardScaler()
        self._process()

    def _process(self):
        all_pampa_values = self.data['PAMPA'].values.reshape(-1, 1)
        self.y_scaler.fit(all_pampa_values)

        all_node_features = []

        for _, row in tqdm(self.data.iterrows(), desc="Processing data"):
            smiles = row['SMILES']
            pampa = row['PAMPA']
            num_residues = row['Num Residues']
            num_residues_in_main_loop = row['Num Residues in Main Cycle']

            peptide = Peptide(smiles, num_residues=num_residues, num_residues_in_main_loop=num_residues_in_main_loop)
            graph_data = peptide.graph_embedding.to_pytorch_geometric()

            all_node_features.append(graph_data.x.numpy())

        # Fit node feature scaler
        self.node_feature_scaler.fit(np.vstack(all_node_features))

        # Transform data
        for i, (_, row) in enumerate(tqdm(self.data.iterrows(), desc="Transforming data")):
            smiles = row['SMILES']
            pampa = row['PAMPA']
            num_residues = row['Num Residues']
            num_residues_in_main_loop = row['Num Residues in Main Cycle']

            peptide = Peptide(smiles, num_residues=num_residues, num_residues_in_main_loop=num_residues_in_main_loop)
            graph_data = peptide.graph_embedding.to_pytorch_geometric()

            # Scale PAMPA value
            scaled_pampa = self.y_scaler.transform([[pampa]])[0][0]
            graph_data.y = torch.tensor([scaled_pampa], dtype=torch.float)

            # Scale node features
            graph_data.x = torch.tensor(self.node_feature_scaler.transform(all_node_features[i]), dtype=torch.float)

            if self.pre_filter is not None and not self.pre_filter(graph_data):
                continue
            if self.pre_transform is not None:
                graph_data = self.pre_transform(graph_data)

            self.processed_data.append(graph_data)

    def len(self):
        return len(self.processed_data)

    def get(self, idx):
        return self.processed_data[idx]

    def clean(self, pampa_lower_bound=-8):
        """Removes peptides with PAMPA values below the specified lower bound."""
        original_pampa_values = self.y_scaler.inverse_transform(np.array([data.y.item() for data in self.processed_data]).reshape(-1, 1))
        self.processed_data = [data for data, original_pampa in zip(self.processed_data, original_pampa_values) if original_pampa[0] >= pampa_lower_bound]
        self.data = self.data[self.data['PAMPA'] >= pampa_lower_bound]
        self.data.reset_index(drop=True, inplace=True)

    def inverse_transform_y(self, y):
        """Inverse transform scaled y values to original scale."""
        return self.y_scaler.inverse_transform(y.reshape(-1, 1)).flatten()
