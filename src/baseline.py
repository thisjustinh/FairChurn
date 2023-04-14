import torch.nn as nn

class BaselineNN(nn.Module):
    """
    Simple binary classification neural network with two hidden layers
    Uses ReLU as activation function for most layers and sigmoid for output layer
    """
    def __init__(self, n_in, n_hidden, n_layers):
        super().__init__()
        
        layers = [nn.Linear(n_in, n_hidden), nn.ReLU()]  # input layer
        for _ in range(n_layers):
            layers.append(nn.Linear(n_hidden, n_hidden))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(n_hidden, 1))  # output layer
        layers.append(nn.Sigmoid())  # output activation
        
        self.model = nn.Sequential(*layers)  # create model from layers list
    
    def forward(self, x):
        return self.model(x)
