import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch import optim
from torch_geometric.data import DataLoader, Data, Batch
from tqdm import tqdm

from ..core.dataset_augmentation import GraphDatasetAugmenter
from helpers import calculate_metrics, print_model_config, plot_metrics_history, plot_true_vs_predicted_and_residuals, EarlyStopping
from modular_graph_attention_transformer import ModularGraphAttentionTransformer


# @title Model Trainer

class ModelTrainer:
    def __init__(self, config):
        self.config = config
        self.device = config['device']
        self.y_scaler = None

    def process_batch(self, batch):
        """Process a batch ensuring all tensors are on the correct device."""
        if not hasattr(batch, 'batch'):
            batch.batch = torch.zeros(batch.x.size(0), dtype=torch.long)

        # List of attributes to check and move to device
        attributes = ['x', 'edge_index', 'edge_attr', 'y', 'batch']

        for attr in attributes:
            if hasattr(batch, attr):
                value = getattr(batch, attr)
                if torch.is_tensor(value):
                    setattr(batch, attr, value.to(self.device))

        return batch

    def prepare_dataset(self, dataset):
        """Split dataset and create dataloaders."""
        # Move dataset to CPU first to ensure consistent starting point

        dataset = [data.cpu() for data in dataset]

        train_val_dataset, test_dataset = train_test_split(dataset, test_size=0.2, random_state=42)
        train_dataset, val_dataset = train_test_split(train_val_dataset, test_size=0.25, random_state=42)

        augmenter = GraphDatasetAugmenter(
            dataset=train_dataset,
            num_bins=20,
            target_number=1000,
            permutations_per_graph=1000,
            scaler = self.y_scaler
        )

        # Visualize the original distribution
        augmenter.plot_distribution(include_augmented=False)

        # Get the augmented dataset
        train_dataset = augmenter.get_augmented_dataset()

        # Visualize the distribution after augmentation
        augmenter.plot_distribution(include_augmented=True)

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            pin_memory=True if torch.cuda.is_available() else False
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            pin_memory=True if torch.cuda.is_available() else False
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            pin_memory=True if torch.cuda.is_available() else False
        )

        return train_loader, val_loader, test_loader

    def unscale_values(self, values):
        """Unscale values using the stored scaler."""
        if self.y_scaler is None:
            raise ValueError("Scaler has not been initialized. Please call y_scaler() first.")

        # Reshape to 2D array if needed (scaler expects 2D input)
        values_2d = np.array(values).reshape(-1, 1)
        unscaled_values = self.y_scaler.inverse_transform(values_2d).flatten()
        return unscaled_values

    def train_epoch(self, model, loader, criterion, optimizer):
        """Train for one epoch."""
        model.train()
        total_loss = 0
        predictions = []
        targets = []

        for batch in loader:
            batch = self.process_batch(batch)
            optimizer.zero_grad()

            out = model(batch)
            loss = criterion(out.squeeze(), batch.y.squeeze())

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch.num_graphs
            predictions.extend(out.squeeze().detach().cpu().numpy())
            targets.extend(batch.y.squeeze().cpu().numpy())

        # Unscale before computing metrics
        unscaled_predictions = self.unscale_values(predictions)
        unscaled_targets = self.unscale_values(targets)
        metrics = calculate_metrics(unscaled_predictions, unscaled_targets)

        return total_loss / len(loader.dataset), metrics, unscaled_predictions, unscaled_targets

    def evaluate(self, model, loader, criterion):
        """Evaluate the model."""
        model.eval()
        total_loss = 0
        predictions = []
        targets = []

        with torch.no_grad():
            for batch in loader:
                batch = self.process_batch(batch)
                out = model(batch)
                loss = criterion(out.squeeze(), batch.y.squeeze())

                total_loss += loss.item() * batch.num_graphs
                predictions.extend(out.squeeze().cpu().numpy())
                targets.extend(batch.y.squeeze().cpu().numpy())

        # Unscale before computing metrics
        unscaled_predictions = self.unscale_values(predictions)
        unscaled_targets = self.unscale_values(targets)
        metrics = calculate_metrics(unscaled_predictions, unscaled_targets)

        return total_loss / len(loader.dataset), metrics, unscaled_predictions, unscaled_targets

    # Rest of the class remains the same...
    def initialize_model(self, node_features, edge_features):
        """Initialize model and move to correct device."""
        model = ModularGraphAttentionTransformer(
            node_features=node_features,
            edge_features=edge_features,
            hidden_channels=self.config['hidden_channels'],
            num_heads=self.config['num_heads'],
            num_layers=self.config['num_layers'],
            dropout=self.config['dropout'],
            attention_type='multihead',
            message_passing='gat',
            pooling_strategy='attention',
            use_edge_features=True
        ).to(self.device)

        print("\nModel initialized on device:", self.device)
        print_model_config(model)
        return model

    def train(self, dataset, node_features, edge_features):
        """Main training loop."""
        self.y_scaler = dataset.y_scaler
        train_loader, val_loader, test_loader = self.prepare_dataset(dataset)
        print(f"\nDataset splits:")
        print(f"Train set size: {len(train_loader.dataset)}")
        print(f"Validation set size: {len(val_loader.dataset)}")
        print(f"Test set size: {len(test_loader.dataset)}")

        model = self.initialize_model(node_features, edge_features)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(
            model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )

        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.config['num_epochs'],
            eta_min=self.config['min_lr'],

        )

        early_stopping = EarlyStopping(patience=20)

        results = self._train_loop(
            model, train_loader, val_loader, test_loader, criterion,
            optimizer, scheduler, early_stopping
        )

        test_loss, test_metrics, test_predictions, test_targets = self.evaluate(
            model, test_loader, criterion
        )

        self.save_model(model, optimizer, scheduler, results, test_metrics)

        return model, results, test_metrics

    def _train_loop(self, model, train_loader, val_loader, test_loader, criterion, optimizer, scheduler, early_stopping):
        """Internal training loop with progress tracking."""
        train_losses = []
        val_losses = []
        train_metrics_history = []
        val_metrics_history = []
        best_val_loss = float('inf')
        best_model_state = None

        epoch_bar = tqdm(range(self.config['num_epochs']), desc='Training', position=0)

        try:
            for epoch in epoch_bar:
                # Training phase
                train_loss, train_metrics, train_preds, train_targets = self.train_epoch(
                    model, train_loader, criterion, optimizer
                )

                # Validation phase
                val_loss, val_metrics, val_preds, val_targets = self.evaluate(
                    model, val_loader, criterion
                )

                # Update learning rate
                scheduler.step()

                # Save metrics
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                train_metrics_history.append(train_metrics)
                val_metrics_history.append(val_metrics)

                # Update progress bar
                epoch_bar.set_postfix({
                    'Train Loss': f'{train_loss:.4f}',
                    'Val Loss': f'{val_loss:.4f}',
                    'Val R²': f'{val_metrics["r2"]:.4f}',
                    'LR': f'{optimizer.param_groups[0]["lr"]:.2e}'
                })

                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = model.state_dict()

                # Early stopping
                early_stopping(val_loss, model)
                if early_stopping.early_stop:
                    print("\nEarly stopping triggered")
                    break

        except Exception as e:
            print(f"\nError during training: {str(e)}")
            raise

        # Load best model
        model.load_state_dict(best_model_state)

        test_loss, test_metrics, test_preds, test_targets = self.evaluate(
            model, test_loader, criterion
        )

        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_metrics_history': train_metrics_history,
            'val_metrics_history': val_metrics_history,
            'val_predictions': val_preds,
            'val_targets': val_targets,
            'test_predictions': test_preds,
            'test_targets': test_targets
        }

    def save_model(self, model, optimizer, scheduler, results, test_metrics):
        """Save model checkpoint."""
        save_path = 'graph_transformer_model.pt'
        try:
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'config': model.config,
                'results': results,
                'test_metrics': test_metrics,
                'scaler': self.scaler  # Save the scaler with the model
            }, save_path)
            print(f"\nModel saved to {save_path}")
        except Exception as e:
            print(f"\nError saving model: {str(e)}")

def main_training_loop(dataset, node_features, edge_features, config):
    """Main entry point for training process."""
    trainer = ModelTrainer(config)
    try:
        model, results, test_metrics = trainer.train(dataset, node_features, edge_features)

        # Plot results (already unscaled in results)
        plot_metrics_history(results)
        plot_true_vs_predicted_and_residuals(
            results['test_targets'],
            results['test_predictions'],
            "Test Set"
        )

        # Print final results
        print("\nTest Set Results:")
        for metric, value in test_metrics.items():
            print(f"{metric.upper()}: {value:.4f}")

        return model, results, test_metrics

    except Exception as e:
        print(f"\nError in training process: {str(e)}")
        raise
