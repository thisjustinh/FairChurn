import torch
import numpy as np
from baseline import BaselineNN
from dataset import ChurnDataset
from torch.utils.data import DataLoader, random_split
from torch.nn import BCELoss
from torch.optim import Adam

import seaborn as sns

def train(n_in, n_hid, lr, batch_size, n_epochs=100, seed=5105):
    # Call dataset for torch use, and get 80-20 train-test split
    ds = ChurnDataset("./data/churn.csv")
    gen = torch.Generator().manual_seed(seed)
    train, test = random_split(ds, [0.8, 0.2], generator=gen)

    # Define data loaders from split
    train_loader = DataLoader(train, shuffle=False, batch_size=batch_size)
    test_loader = DataLoader(train, shuffle=False)

    # Model, loss, optimizer
    model = BaselineNN(n_in, n_hid)
    bce = BCELoss()
    optimizer = Adam(model.parameters(), lr=lr)
    
    losses = []
    for i in range(n_epochs):
        epoch_loss = []
        for f, l in train_loader:
            # forward pass
            out = model(f)
            loss = bce(out, l)
            epoch_loss.append(loss.item())
            # backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        # progress
        losses.append(np.mean(epoch_loss))
        acc = (out.round() == l).float().mean()
        print(f"{i}: {acc}")

    # Save model parameters
    torch.save(model.state_dict(), "./models/baseline.pt")

    # Get test accuracy
    accs = []
    for f, l in test_loader:
        model.eval()
        out = model(f)
        accs.append((out.round() == l).float().mean().item())
    
    return sum(accs) / len(accs), losses

if __name__ == '__main__':
    # Hyperparameters
    n_in = 12
    n_hid = 48
    lr = 0.01
    batch_size=128
    n_epochs = 100

    test_acc, losses = train(n_in, n_hid, lr, batch_size, n_epochs=n_epochs)
    print(test_acc)
    losses = sns.lineplot(x=list(range(0, len(losses))), y=losses)
    losses.get_figure().savefig("./models/baseline_loss.png")