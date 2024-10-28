# @title Data Augmenter

from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch_geometric.data import Data

class GraphDatasetAugmenter:
    def __init__(self, dataset, num_bins=20, target_number=1000, permutations_per_graph=1000, scaler=None):
        """
        Initialize the augmenter with either a PeptideDataset instance or a list of PyG Data objects.

        Args:
            dataset: Either a PeptideDataset instance or a list of PyG Data objects
            num_bins: Number of bins to divide the PAMPA value range into
            permutations_per_graph: Base number of permutations per graph
            scaler: Optional scaler for inverse transform of y values
        """
        self.dataset = dataset
        self.num_bins = num_bins
        self.permutations_per_graph = permutations_per_graph
        self.y_scaler = scaler

        # Handle different dataset types
        if hasattr(dataset, 'processed_data'):
            self.data_list = dataset.processed_data
        else:
            self.data_list = dataset

        # Get original PAMPA values (unscaled)
        self.pampa_values = []
        for data in self.data_list:
            y_value = data.y.cpu().numpy()
            if self.y_scaler is not None:
                # Reshape for scaler which expects 2D input
                y_value = y_value.reshape(-1, 1)
                y_value = self.y_scaler.inverse_transform(y_value).flatten()
            self.pampa_values.append(y_value[0])
        self.pampa_values = np.array(self.pampa_values)

        # Create bins using unscaled values
        self.bins = np.linspace(
            self.pampa_values.min(),
            self.pampa_values.max(),
            num_bins + 1
        )
        self.bin_indices = np.digitize(self.pampa_values, self.bins) - 1

        # Count samples in each bin
        self.bin_counts = np.bincount(self.bin_indices, minlength=num_bins)

        # Set target count to the maximum bin count
        self.target_count = np.max([int(np.max(self.bin_counts)), target_number])

    def plot_distribution(self, include_augmented=True):
        """
        Plot the distribution of PAMPA values before and after augmentation.
        All values are plotted in their original unscaled form.
        """
        plt.figure(figsize=(9, 4))

        if include_augmented:
            augmented_data = self.augment()
            augmented_values = []

            # Unscale augmented values
            for data in augmented_data:
                y_value = data.y.cpu().numpy()
                if self.y_scaler is not None:
                    y_value = y_value.reshape(-1, 1)
                    y_value = self.y_scaler.inverse_transform(y_value).flatten()
                augmented_values.append(y_value[0])
            augmented_values = np.array(augmented_values)

            # Plot augmented first (so original is on top)
            plt.hist(augmented_values, bins=self.bins, alpha=0.5,
                    label='Augmented', color='orange')

            # Print statistics using unscaled values
            print("\nBin Statistics (Original Scale):")
            hist, _ = np.histogram(augmented_values, bins=self.bins)
            for i, (count, bin_edge) in enumerate(zip(hist, self.bins[:-1])):
                print(f"Bin {i} ({bin_edge:.2f} to {self.bins[i+1]:.2f}): {count} samples")

        # Plot original values (already unscaled in self.pampa_values)
        plt.hist(self.pampa_values, bins=self.bins, alpha=0.5,
                label='Original', color='skyblue')

        plt.xlabel('PAMPA Value (Original Scale)')
        plt.ylabel('Frequency')
        plt.title('Distribution of PAMPA Values')
        plt.legend()
        plt.show()

        # Print additional statistics
        print("\nDistribution Statistics (Original Scale):")
        print(f"Original data range: {self.pampa_values.min():.2f} to {self.pampa_values.max():.2f}")
        if include_augmented:
            print(f"Augmented data range: {augmented_values.min():.2f} to {augmented_values.max():.2f}")
            print(f"Total samples - Original: {len(self.pampa_values)}, Augmented: {len(augmented_values)}")

    def permute_graph(self, graph_data):
        """
        Create a permuted version of the input graph while preserving its structure and edge attributes.
        """
        device = graph_data.x.device
        num_nodes = graph_data.x.size(0)

        # Generate random permutation
        perm = torch.randperm(num_nodes, device=device)

        # Create inverse permutation mapping
        inv_perm = torch.zeros_like(perm)
        inv_perm[perm] = torch.arange(num_nodes, device=device)

        # Permute node features and edge indices
        new_x = graph_data.x[perm]
        new_edge_index = graph_data.edge_index.clone()
        new_edge_index = inv_perm[new_edge_index]

        # Create new graph with all attributes
        new_data = {
            'x': new_x,
            'edge_index': new_edge_index,
            'y': graph_data.y.clone()
        }

        # Copy edge attributes if they exist
        if hasattr(graph_data, 'edge_attr'):
            new_data['edge_attr'] = graph_data.edge_attr.clone()

        # Copy any additional attributes from the original graph
        for key in graph_data.keys():
            if key not in ['x', 'edge_index', 'edge_attr', 'y']:
                new_data[key] = graph_data[key]

        return Data(**new_data)

    def augment(self):
        """
        Perform dataset augmentation to match all bins to the size of the largest bin.
        """
        augmented_data = []

        # Group original data by bins
        bin_to_graphs = defaultdict(list)
        for idx, bin_idx in enumerate(self.bin_indices):
            bin_to_graphs[bin_idx].append(self.data_list[idx])

        # Find non-empty bins
        non_empty_bins = [i for i in range(self.num_bins) if self.bin_counts[i] > 0]

        # For each non-empty bin that needs augmentation
        for bin_idx in non_empty_bins:
            current_count = self.bin_counts[bin_idx]
            samples_needed = self.target_count - current_count

            if samples_needed <= 0:
                continue

            graphs_in_bin = bin_to_graphs[bin_idx]

            # Calculate permutations needed per graph
            perms_per_graph = int(np.ceil(samples_needed / len(graphs_in_bin)))

            # Generate samples
            samples_generated = 0
            while samples_generated < samples_needed:
                for graph in graphs_in_bin:
                    if samples_generated >= samples_needed:
                        break

                    augmented_data.append(self.permute_graph(graph))
                    samples_generated += 1

                    # If we've used all graphs but still need more samples,
                    # start over with the same graphs
                    if samples_generated < samples_needed and graph == graphs_in_bin[-1]:
                        continue
        augmented_data = augmented_data + self.data_list
        return augmented_data

    def get_augmented_dataset(self):
        """
        Returns a new dataset containing both original and augmented samples.
        """
        augmented_data = self.augment()
        return self.data_list + augmented_data
