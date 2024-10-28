# @title Helpers

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import torch
from torch_geometric.data import Data, DataLoader

from modular_graph_attention_transformer import ModularGraphAttentionTransformer

class EarlyStopping:
    def __init__(self, patience=7, min_delta=0, verbose=False):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

def plot_learning_curves(train_losses, val_losses, train_mses, val_mses):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Plot losses
    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(val_losses, label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Losses')
    ax1.legend()

    # Plot MSEs
    ax2.plot(train_mses, label='Train MSE')
    ax2.plot(val_mses, label='Validation MSE')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('MSE')
    ax2.set_title('Training and Validation MSE')
    ax2.legend()

    plt.tight_layout()
    plt.show()

def plot_true_vs_predicted_and_residuals(targets, predictions, title=""):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # True vs Predicted
    ax1.scatter(targets, predictions, alpha=0.5)
    ax1.plot([min(targets), max(targets)], [min(targets), max(targets)], 'r--')
    ax1.set_xlabel('True Values')
    ax1.set_ylabel('Predicted Values')
    ax1.set_title(f'{title} - True vs Predicted')

    # Residuals
    residuals = np.array(predictions) - np.array(targets)
    ax2.scatter(predictions, residuals, alpha=0.5)
    ax2.axhline(y=0, color='r', linestyle='--')
    ax2.set_xlabel('Predicted Values')
    ax2.set_ylabel('Residuals')
    ax2.set_title(f'{title} - Residuals Plot')

    plt.tight_layout()
    plt.show()

def calculate_metrics(predictions, targets):
    """Calculate multiple regression metrics."""
    predictions = np.array(predictions)
    targets = np.array(targets)

    mse = np.mean((predictions - targets) ** 2)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(targets, predictions)
    r2 = r2_score(targets, predictions)

    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    predictions = []
    targets = []

    for batch in loader:
        # Move entire batch to device
        batch = batch.to(device)
        optimizer.zero_grad()

        # Forward pass
        out = model(batch)
        loss = criterion(out.squeeze(), batch.y.squeeze())

        # Backward pass
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch.num_graphs
        predictions.extend(out.squeeze().detach().cpu().numpy())
        targets.extend(batch.y.squeeze().cpu().numpy())

    metrics = calculate_metrics(predictions, targets)
    return total_loss / len(loader.dataset), metrics, predictions, targets

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    predictions = []
    targets = []

    with torch.no_grad():
        for batch in loader:
            # Move entire batch to device
            batch = batch.to(device)
            out = model(batch)
            loss = criterion(out.squeeze(), batch.y.squeeze())

            total_loss += loss.item() * batch.num_graphs
            predictions.extend(out.squeeze().cpu().numpy())
            targets.extend(batch.y.squeeze().cpu().numpy())

    metrics = calculate_metrics(predictions, targets)
    return total_loss / len(loader.dataset), metrics, predictions, targets

def setup_training(dataset, node_features, edge_features, config):
    """Setup all training components."""
    # Split dataset
    train_val_dataset, test_dataset = train_test_split(dataset, test_size=0.2, random_state=42)
    train_dataset, val_dataset = train_test_split(train_val_dataset, test_size=0.3, random_state=42)

    print(f"\nDataset splits:")
    print(f"Train set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")
    print(f"Test set size: {len(test_dataset)}")
    print(f"\nFeatures:")
    print(f"Number of node features: {node_features}")
    print(f"Number of edge features: {edge_features}")

    # Setup device first
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    # Create dataloaders with pin_memory for faster data transfer to GPU
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'],
                            shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'],
                          pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'],
                           pin_memory=True)

    # Initialize model
    model = ModularGraphAttentionTransformer(
        node_features=node_features,
        edge_features=edge_features,
        hidden_channels=config['hidden_channels'],
        num_heads=config['num_heads'],
        num_layers=config['num_layers'],
        dropout=config['dropout'],
        attention_type='multihead',
        message_passing='gat',
        pooling_strategy='attention',
        use_edge_features=True
    ).to(device)  # Move model to device immediately after initialization

    # Print model configuration
    print_model_config(model)

    # Setup training components
    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=10,
        min_lr=config['min_lr']
    )

    early_stopping = EarlyStopping(patience=20)

    return {
        'model': model,
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'criterion': criterion,
        'optimizer': optimizer,
        'scheduler': scheduler,
        'early_stopping': early_stopping,
        'device': device
    }

def plot_metrics_history(history):
    """Plot detailed training history including all metrics."""
    metrics = ['mse', 'rmse', 'mae', 'r2']
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.ravel()

    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        train_metric = [epoch_metrics[metric] for epoch_metrics in history['train_metrics_history']]
        val_metric = [epoch_metrics[metric] for epoch_metrics in history['val_metrics_history']]

        ax.plot(train_metric, label=f'Train {metric.upper()}')
        ax.plot(val_metric, label=f'Val {metric.upper()}')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric.upper())
        ax.set_title(f'{metric.upper()} over Training')
        ax.legend()

    plt.tight_layout()
    plt.show()

def print_model_config(model):
    """Print model configuration and parameters."""
    print("\nModel Configuration:")
    for key, value in model.config.items():
        print(f"{key}: {value}")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nTotal Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Model Size: {total_params * 4 / 1024 / 1024:.2f} MB")

def main_training_loop(dataset, node_features, edge_features, config):
    """Main training function that orchestrates the entire training process."""

    # Setup all components
    training_setup = setup_training(dataset, node_features, edge_features, config)

    # Train model
    results = train_model(
        model=training_setup['model'],
        train_loader=training_setup['train_loader'],
        val_loader=training_setup['val_loader'],
        criterion=training_setup['criterion'],
        optimizer=training_setup['optimizer'],
        scheduler=training_setup['scheduler'],
        device=training_setup['device'],
        early_stopping=training_setup['early_stopping']
    )

    # Evaluate on test set
    test_loss, test_metrics, test_predictions, test_targets = evaluate(
        model=training_setup['model'],
        loader=training_setup['test_loader'],
        criterion=training_setup['criterion'],
        device=training_setup['device']
    )

    # Print final results
    print("\nTest Set Results:")
    for metric, value in test_metrics.items():
        print(f"{metric.upper()}: {value:.4f}")

    # Plot results
    plot_metrics_history(results)
    plot_true_vs_predicted_and_residuals(
        results['val_targets'],
        results['val_predictions'],
        "Validation Set"
    )
    plot_true_vs_predicted_and_residuals(
        test_targets,
        test_predictions,
        "Test Set"
    )

    # Save model
    save_path = 'graph_transformer_model.pt'
    torch.save({
        'model_state_dict': training_setup['model'].state_dict(),
        'optimizer_state_dict': training_setup['optimizer'].state_dict(),
        'scheduler_state_dict': training_setup['scheduler'].state_dict(),
        'config': training_setup['model'].config,
        'results': results,
        'test_metrics': test_metrics
    }, save_path)
    print(f"\nModel saved to {save_path}")

    return training_setup['model'], results, test_metrics


def create_model(input_dim, edge_dim, hidden_channels=256, num_heads=8, num_layers=6):
    # Make sure edge_dim is correctly passed through
    edge_features = edge_dim if edge_dim is not None else 0

    model = ModularGraphAttentionTransformer(
        node_features=input_dim,
        edge_features=edge_features,
        hidden_channels=hidden_channels,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=0.1,
        attention_type='multihead',
        message_passing='gat',
        pooling_strategy='attention',
        activation='gelu',
        norm_type='batch',
        residual_mode='dense',
        position_encoding=True,
        pe_type='laplacian',
        num_pe_features=16,
        walk_length=8,
        is_undirected=True,
        edge_dim=edge_features,  # Pass the actual edge dimension
        virtual_nodes=False,
        ffn_ratio=4,
        pooling_ratio=0.5,
        use_edge_features=edge_features > 0,
        residual_features=None
    )
    return model

def evaluate_model(model, loader, criterion, device, dataset):
    model.eval()
    total_loss = 0
    total_mse = 0
    predictions = []
    targets = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch).squeeze()
            loss = criterion(out, batch.y.squeeze())
            total_loss += loss.item() * batch.num_graphs

            # Unscale predictions and targets for MSE calculation
            unscaled_out = dataset.inverse_transform_y(out.cpu().numpy())
            unscaled_y = dataset.inverse_transform_y(batch.y.squeeze().cpu().numpy())
            total_mse += mean_squared_error(unscaled_y, unscaled_out) * batch.num_graphs

            predictions.extend(unscaled_out)
            targets.extend(unscaled_y)

    avg_loss = total_loss / len(loader.dataset)
    avg_mse = total_mse / len(loader.dataset)
    r2 = r2_score(targets, predictions)

    return avg_loss, avg_mse, r2, np.array(predictions), np.array(targets)
