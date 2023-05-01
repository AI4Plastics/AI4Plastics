import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear
from torch.nn import BatchNorm1d
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_add_pool

class GCN(nn.Module):
    def __init__(self, n_convolutions, convolutions_dim, n_hidden_layers, hidden_layers_dim):
        super(GCN, self).__init__()
        
        self.heatmaps = False
        self.gradients = 0
        
        self.n_convolutions = n_convolutions
        self.n_hidden_layers = n_hidden_layers

        # Configurable pipeline
        self.layers = nn.ModuleList()

        for i in range(n_convolutions):
            self.layers.append(GCNConv(-1 if i == 0 else convolutions_dim, convolutions_dim))
            self.layers.append(BatchNorm1d(convolutions_dim))

        for i in range(n_hidden_layers):
            self.layers.append(Linear(convolutions_dim if i == 0 else hidden_layers_dim, hidden_layers_dim))
            self.layers.append(BatchNorm1d(hidden_layers_dim))

        self.layers.append(Linear(hidden_layers_dim, 1))
        
    def activations_hook(self, gradient):
        self.gradients = gradient
    
    def get_gradients(self):
        return self.gradients
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        heatmap_activations = 0

        n_conv_batch = 2 * self.n_convolutions
        n_hidden_batch = 2 * self.n_hidden_layers

        # Forward pass for convolutional head
        for idx, layer in enumerate(self.layers[:n_conv_batch]):
            # Wrap convolutions (every other layer) in relu
            if idx % 2 == 0:
                x = F.relu(layer(x, edge_index))
                print(n_conv_batch)
                if self.heatmaps and idx == n_conv_batch-2:
                    x.register_hook(self.activations_hook)
                    heatmap_activations = x
                    
            # No relu for batch norm
            else:
                x = layer(x)

        # After convolutional head, apply global pooling
        x = global_add_pool(x, data.batch)

        # Forward pass for multilayer preceptron
        for idx, layer in enumerate(self.layers[n_conv_batch:]):
            # Wrap dense connections (every other layer) in relu
            if idx % 2 == 0 and idx != n_hidden_batch:
                x = F.relu(layer(x))
            
            # No relu for batch norm or output
            else:
                x = layer(x)

        return x