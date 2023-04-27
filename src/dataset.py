import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler


class ChurnDataset(Dataset):
    """
    Custom PyTorch Dataset class for use with Pytorch DataLoader.
    Preprocessing is completed in the constructor, then the generator returns tuple of features, labels, and sens.
    """
    def __init__(self, filepath):
        # Import data from CSV
        df = pd.read_csv(filepath)

        # Clean data: drop unnecessary cols, encode Geography 
        cleaned = df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)
        one_hot_geo = pd.get_dummies(cleaned['Geography'])
        cleaned = cleaned.drop('Geography', axis=1)
        cleaned = one_hot_geo.join(cleaned)

        # Clean data: encode gender to binary (Male = 0, Female = 1)
        cleaned['Gender'].replace(('Male', 'Female'), (0, 1), inplace=True)

        # Log transform EstimatedSalary and Balance
        cleaned['EstimatedSalary'] = np.log(cleaned.pop('EstimatedSalary') + 1)
        cleaned['Balance'] = np.log(cleaned.pop('Balance') + 1)

        # Define sensitive attribute, here it's age
        age = pd.Series(np.where(cleaned['Age'] < np.median(cleaned['Age']), 0, 1), cleaned.index)
        # print(np.median(cleaned['Age']))

        # Split into features and labels
        y = cleaned.pop('Exited')
        scaler = StandardScaler()
        x = scaler.fit_transform(cleaned)  # features are mean 0 and sd 1

        # Features, Labels, Sensitive Attribute as tensors
        self.x = torch.tensor(x, dtype=torch.float32)  # size = [nrow, ncol - 1]
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)  # size = [nrow, 1]
        self.sens = torch.tensor(age, dtype=torch.float32).unsqueeze(1)  # size = [nrow, 1]

    def shape(self):
        return self.x.shape, self.y.shape
    
    def get_labels(self):
        return self.y

    def __getitem__(self, index):
        return (
            self.x[index],  # features
            self.y[index],  # labels
            self.sens[index]  # sensitive attributes
        )

    def __len__(self):
        return len(self.x)
