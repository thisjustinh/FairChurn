import torch
import torch.optim as optim
import numpy as np
from baseline import BaselineNN
from dataset import ChurnDataset
from torch.utils.data import DataLoader, WeightedRandomSampler, random_split
from torch.nn import BCEWithLogitsLoss
from sklearn.metrics import roc_auc_score

import seaborn as sns
import matplotlib.pyplot as plt

def train(n_in, n_hid, n_layers, lr, batch_size, n_epochs=100, weight=1, init_b=False, seed=1):
    # Call dataset for torch use, and get 80-20 train-test split
    ds = ChurnDataset("./data/churn.csv")
    gen = torch.Generator().manual_seed(seed)
    train, test = random_split(ds, [0.7, 0.3], generator=gen)

    # Create weights for weighted sampler, to get even 0/1 cases from batch
    y_train = [ds.get_labels()[i].item() for i in train.indices]
    class_sample_count = np.array([len(np.where(y_train == t)[0]) for t in np.unique(y_train)])
    weights = 1. / class_sample_count
    samples_weight = torch.tensor([weights[int(i)] for i in y_train], dtype=torch.float32)
    strat_sampler = WeightedRandomSampler(samples_weight, len(samples_weight), generator=gen)

    # Define data loaders from split
    train_loader = DataLoader(train, sampler=strat_sampler, shuffle=False, batch_size=batch_size)
    test_loader = DataLoader(test, shuffle=False)

    # Model, loss, optimizer
    model = BaselineNN(n_in, n_hid, n_layers)
    if init_b:  # initialize bias for imbalanced labels
        model.apply(init_bias)
    
    print(class_sample_count[1]/class_sample_count[0])
    bce = BCEWithLogitsLoss(pos_weight=torch.tensor(weight*class_sample_count[0]/class_sample_count[1], dtype=torch.float32))
    # bce = BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_auc = 0
    best_model_params = None
    
    losses = []
    for i in range(n_epochs):
        epoch_loss = []
        for f, l in train_loader:
            # forward pass
            logit = model(f)  
            out = torch.sigmoid(logit)  # since I am using BCEWithLogitsLoss and don't have sigmoid at end of neural network
            loss = bce(logit, l)
            epoch_loss.append(loss.item())
            # backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        # progress
        losses.append(np.mean(epoch_loss))
        acc = (out.round() == l).float().mean()
        auc = roc_auc_score(l.detach().cpu().numpy(), out.detach().cpu().numpy())
        print(f"{i}: accuracy {acc}, AUC {auc}")

        # save best model by auc
        if auc >= best_auc:  # we prefer higher epochs of training, so >=
            best_auc = auc
            best_model_params = model.state_dict()

    # Save model parameters
    torch.save(model.state_dict(), "./models/baseline.pt")
    torch.save(best_model_params, "./models/high_auc_baseline.pt")

    print(f"Best train AUC: {best_auc}")
    
    return losses, test_loader


def run_test_metrics(n_in, n_hidden, n_layers, path, loader):
    model = BaselineNN(n_in, n_hidden, n_layers)
    model.load_state_dict(torch.load(path))
    model.eval()

    gt = np.array([])
    pred = np.array([])

    tp = 0  # true positive
    fp = 0  # false positive
    fn = 0  # false negative
    tn = 0  # true negative

    for feats, labels in loader:  # evaluate test set
        out = torch.sigmoid(model(feats))
        gt = np.append(gt, labels.detach().cpu().numpy())
        pred = np.append(pred, out.detach().cpu().numpy())

        for i, l in enumerate(labels):  # check which values are true or not
            single_pred = out[i].round()
            # print(single_pred, 1 - single_pred)
            if l == single_pred:
                if l == 1:
                    tp += 1
                else:
                    tn += 1
            else:
                if l == 1:
                    fn += 1
                else:
                    fp += 1

    gt = gt.flatten()
    pred = pred.flatten()

    conf_mat = sns.heatmap([[tn, fn], [fp, tp]], annot=True)  # plot confusion matrix
    conf_mat.set(xlabel="Ground Truth", ylabel="Predicted")
    conf_mat.get_figure().savefig("./models/baseline_confusion_matrix.png", dpi=300, bbox_inches='tight')

    acc = (tp + tn) / (tp + fp + fn + tn)  # test accuracy
    auc = roc_auc_score(gt, pred)
    return acc, auc


def init_bias(m):
    """
    Initialize bias values for neural network
    The bias number is hard coded for PyTorch to call apply, and the number is log(pos_labels/neg_labels)
    """
    if isinstance(m, torch.nn.Linear):
        m.bias.data.fill_(np.exp(0.2524601896582573))
        

if __name__ == '__main__':
    # Hyperparameters
    n_in = 12
    n_hid = 12
    n_layers = 2
    lr = 0.0001
    batch_size=50
    n_epochs = 100
    init_b = True
    seed = 5105
    weight = 0.5

    losses, test_loader = train(n_in, 
                                n_hid, 
                                n_layers, 
                                lr, 
                                batch_size, 
                                n_epochs=n_epochs,
                                weight=weight, 
                                init_b=init_b, 
                                seed=seed)
    test_acc = run_test_metrics(n_in, n_hid, n_layers, "./models/high_auc_baseline.pt", test_loader)
    print(test_acc)
    plt.clf()
    loss_fig = sns.lineplot(x=list(range(0, len(losses))), y=losses)
    loss_fig.set(xlabel="Epoch", ylabel="BCE Loss")
    loss_fig.get_figure().savefig("./models/baseline_loss.png", dpi=300, bbox_inches='tight')