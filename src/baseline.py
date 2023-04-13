import torch.nn as nn

class BaselineNN(nn.Module):
    """
    Simple binary classification neural network with two hidden layers
    Uses ReLU as activation function for most layers and sigmoid for output layer
    """
    def __init__(self, n_in, n_hidden):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(n_in, n_hidden),  # input
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),  # first hidden
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),  # second hidden
            nn.ReLU(),
            nn.Linear(n_hidden, 1),  # output
            nn.Sigmoid(),
        )
    
    def forward(self, x):
        return self.model(x)
