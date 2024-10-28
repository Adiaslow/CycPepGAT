# @title Train Model

import torch
from .helpers import main_training_loop

config = {
    'batch_size': 16,
    'hidden_channels': 256,
    'num_heads': 8,
    'num_layers': 6,
    'dropout': 0.2,
    'learning_rate': 0.001,
    'weight_decay': 1e-6,
    'num_epochs': 100,
    'min_lr': 1e-7,
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'scaler': dataset
}

node_features = 37
edge_features = 17

model, results, test_metrics = main_training_loop(
    dataset=dataset,
    node_features=node_features,
    edge_features=edge_features,
    config=config
)

# 4. Make predictions with trained model
def predict(model, data, device):
    model.eval()
    with torch.no_grad():
        data = data.to(device)
        prediction = model(data)
    return prediction.cpu().numpy()

# Example prediction on a single graph
example_graph = dataset[0]
prediction = predict(model, example_graph, device='cuda' if torch.cuda.is_available() else 'cpu')

# 5. Save and load model
# Save
def save_model(model, path='model_checkpoint.pt'):
    """Save model with its configuration."""
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': {
            'node_features': model.config['node_features'],
            'edge_features': model.config['edge_features'],
            'hidden_channels': model.config['hidden_channels'],
            'num_heads': model.config['num_heads'],
            'num_layers': model.config['num_layers'],
            'dropout': model.config['dropout']
        }
    }, path)

def load_model(checkpoint_path):
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path)
    config = checkpoint['config']

    model.load_state_dict(checkpoint['model_state_dict'])
    return model

save_model(model, 'model_checkpoint.pt')

loaded_model = load_model('model_checkpoint.pt')
