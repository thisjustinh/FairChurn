# FairChurn
Our goal is to predict bank customer churn while mitigating bias on age, since a client's age shouldn't be a direct indicator on why they may or may not leave. We consider a baseline neural network and a debiased neural network and compare results.

The deep learning implementation is done completely in PyTorch. The Conda environment used can be found in the `environment.yml` file.

## Classes

We create three classes: `ChurnDataset`, `BaselineNN`, and `FairAdversary`. All three can be found in their own Python files under the `src` folder.

### ChurnDataset

This class inherits from the `torch.utils.data.Dataset` class to be used in PyTorch DataLoaders. Here, we perform all relevant preprocessing for the given dataset in `data/churn.csv`. This includes dropping unusable columns like Surname and Row Number, redefining gender as a binary integer column, creating one-hot columns for the categorical Geography column, and log-transforming imbalanced numerical columns (Balance and EstimatedSalary). Finally, we store the following variables which are given in every batch of training:

- `x`: The features from the dataset, scaled to mean 0 and standard deviation 1.
- `y`: The labels (binary, 0 for staying and 1 for exiting)
- `sens`: The sensitive attribute (binary flag, 0 for age below median and 1 for above)

### BaselineNN

The baseline neural network is flexibly created based on the number of hidden layers desired, but generally, the layout is a linear layer followed by a ReLU activation. The final layer is a linear layer that results in the output logits. We don't include an activation function here (e.g. sigmoid) because our loss function, `torch.nn.BCEWithLogitsLoss` includes the sigmoid transformation already. 

### FairChurn

The process of debiasing involves a zero-sum game between the classifier and an adversarial discriminator trying to determine the sensitive attribute given the predicted label from the classifier. As such, we define an adversarial classifier that takes in a certain number of inputs, includes two hidden layers followed by ReLU activation, and then has an output layer with one node (for the sensitive attribute).

## Training and Testing

To run the models, the baseline model and FairChurn model use `run_baseline.py` and `run_fairchurn.py`, respectively. We describe the process of training and testing below, for both.

### The Baseline

To train, we create the ChurnDataset, split into a 50-50 train-test split, and set those into DataLoaders. Importantly, we add a WeightedRandomSampler to the train DataLoader so each batch has an equal number of positive and negative examples, since the dataset is imbalanced (about 80% of observations are customers staying).

We use `BCEWithLogitsLoss` as the loss function because, by defining positive weights, we can weigh the loss based on the imbalance of positive and negative samples from the dataset and make predictions more balanced as a result. Adam is used for optimization. This functionality does not exist for normal binary cross-entropy in PyTorch.

For every epoch, we iterate through all batches and run the standard forward and backward passes, just like any other neural network. The losses are recorded, and are returned along with the test set DataLoader.

To evaluate performance on said DataLoader, we iterate through and record true positive, true negatives, false positives, and false negatives (overall values, values for the protected group, and values for the unprotected group). These values are used to generate a confusion matrix and for calculating fairness metrics like demographic parity, equal opportunity, and equalized odds.

### FairChurn

Running FairChurn uses almost the exact same process for both training and testing, but with a few differences in training. For each epoch, we alternate updating the adversary and updating the classifier. The adversary update also uses `BCEWithLogitsLoss`.

For the classifier, our composite loss becomes `classifier_loss - adversary_loss` since we want the adversary to perform poorly, so the classifier is penalized for low adversary loss and rewarded for high adversary loss. In other words, the classifier needs to trick the adversary so it can't predict the sensitive attribute. 

## Best Performance

Here, we include relevant hyperparameters used for both the baseline and FairChurn neural networks. Relevant graphs of performance for both, as well as pretrained model files, can be found under the `models` folder.

The following hyperparameters are defined:

- `n_in`: Number of inputs for classifier
- `n_hid`: Number of hidden nodes
- `n_layers`: Number of hidden layers for classifier
- `lr`: Learning rate for classifier & adversary
- `batch_size`: How many training examples per model update
- `n_epochs`: Number of training epochs
- `init_b`: Whether to initialize bias for classifier or not
- `seed`: Random seed for reproducibility
- `weight`: Scales how strong the positive weight is for the loss function
- `adv_weight`: How much we care about fairness (< 1 prioritize performance, > 1 prioritize fairness)

### Baseline NN

```{python3}
n_in = 12
n_hid = 12
n_layers = 2
lr = 0.0001
batch_size=50
n_epochs = 1000
init_b = True
seed = 5105
weight = 0.1
```

- Best train AUC: 1.0
- Test accuracy: 0.847, AUC: 0.8633317273659533
- Demographic Parity: -0.30260182514488904
- Equal Opportunity: -0.19996830972676538
- Equalized Odds FPR: -0.10263351541812361

### FairChurn

#### Best for Parity

```{python3}
n_in = 12
n_hid = 12
n_layers = 2
lr = 0.0001
batch_size=50
n_epochs = 1000
init_b = True
seed = 5105
weight = 0.15
adv_weight = 2
```

- Best train AUC: 0.9915110356536503
- Test accuracy: 0.7856666666666666, AUC: 0.8065888176222171
- Demographic Parity: -0.028131831536655966
- Equal Opportunity: -0.13909075771980428
- Equalized Odds FPR: 0.1109589261831483

#### Best for Equalized Odds

```{python3}
n_in = 12
n_hid = 12
n_layers = 2
lr = 0.0001
batch_size=50
n_epochs = 1000
init_b = True
seed = 5105
weight = 0.1
adv_weight = 1
```

- Best train AUC: 0.9830220713073005
- Test accuracy: 0.8136666666666666, AUC: 0.8069811132852979
- Demographic Parity: -0.08643119182283526
- Equal Opportunity: -0.11148799260918373
- Equalized Odds FPR: 0.025056800786348477

### Statistic Summary
- The Mean:
 CreditScore           650.528800
Age                    38.921800
Tenure                  5.012800
Balance             76485.889288
IsActiveMember          0.515100
EstimatedSalary    100090.239881
Exited                  0.203700

The Median:
 CreditScore           652.000
Age                    37.000
Tenure                  5.000
Balance             97198.540
IsActiveMember          1.000
EstimatedSalary    100193.915
Exited                  0.000

The Mode:
 CreditScore          850.00
Age                   37.00
Tenure                 2.00
Balance                0.00
IsActiveMember         1.00
EstimatedSalary    24924.92
Exited                 0.00

The Variance:
 CreditScore        9.341860e+03
Age                1.099941e+02
Tenure             8.364673e+00
Balance            3.893436e+09
IsActiveMember     2.497970e-01
EstimatedSalary    3.307457e+09
Exited             1.622225e-01

The Standard Deviation:
 CreditScore           96.653299
Age                   10.487806
Tenure                 2.892174
Balance            62397.405202
IsActiveMember         0.499797
EstimatedSalary    57510.492818
Exited                 0.402769

The Minimum:
 CreditScore        350.00
Age                 18.00
Tenure               0.00
Balance              0.00
IsActiveMember       0.00
EstimatedSalary     11.58
Exited               0.00

The Maximum:
 CreditScore           850.00
Age                    92.00
Tenure                 10.00
Balance            250898.09
IsActiveMember          1.00
EstimatedSalary    199992.48
Exited                  1.00

The Range:
 CreditScore           500.00
Age                    74.00
Tenure                 10.00
Balance            250898.09
IsActiveMember          1.00
EstimatedSalary    199980.90
Exited                  1.00

The Maximum Likelihood:
 CreditScore           850.00
Age                    92.00
Tenure                 10.00
Balance            250898.09
IsActiveMember          1.00
EstimatedSalary    199992.48
Exited                  1.00