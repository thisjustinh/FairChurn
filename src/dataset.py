import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler


class ChurnDataset(Dataset):

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

        print(cleaned.head())

        # Split into features and labels
        y = cleaned.pop('Exited')
        scaler = StandardScaler()
        x = scaler.fit_transform(cleaned)  # features are mean 0 and sd 1

        # Features, Labels as tensors
        self.x = torch.tensor(x, dtype=torch.float32)  # size = [nrow, ncol - 1]
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)  # size = [nrow, 1]

    def shape(self):
        return self.x.shape, self.y.shape
    
    def get_labels(self):
        return self.y

    def __getitem__(self, index):
        # return {
        #     'feature': torch.tensor([self.x[index]], dtype=torch.float32),
        #     'label': torch.tensor([self.y[index]], dtype=torch.float32)
        # }
        return (
            self.x[index],
            self.y[index]
        )

    def __len__(self):
        return len(self.x)