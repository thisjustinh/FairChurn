import torch
import pandas as pd
from torch.utils.data import Dataset


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

        # Turn into tensors
        x = cleaned.iloc[:, :-1]
        y = cleaned.iloc[:, -1]

        # Features, Labels
        self.x = torch.tensor(x.values, dtype=torch.float32)  # size = [nrow, ncol - 1]
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)  # size = [nrow, 1]

    def shape(self):
        return self.x.shape, self.y.shape

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