import torch.nn as nn

class FairAdversary(nn.Module):
    """
    Create adversary for debiasing as proposed in Zhang et al.
    Adversary seeks to predict the sensitive attribute given the classifier output
    """
    def __init__(self, n_out, n_hidden):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(1, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_out),
        )
    
    def forward(self, x):
        return self.model(x)
