import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_size, units, activation):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        in_dim = input_size
        if activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'relu':
            self.activation = torch.relu
        elif activation == 'sigmoid':
            self.activation = torch.sigmoid
        elif activation == 'elu':
            self.activation = torch.nn.functional.elu
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        for out_dim in units:
            layer = nn.Linear(int(in_dim), out_dim)
            self.layers.append(layer)
            self.init_weights(layer)
            in_dim = out_dim
        
    def init_weights(self, layer):
        """Apply default initialization to a layer."""
        if isinstance(layer, nn.Linear):
            # Use Xavier initialization for weights
            nn.Identity(layer.weight)
            # Initialize biases to zero
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        for layer in self.layers:
            x = self.activation(layer(x))
        return x