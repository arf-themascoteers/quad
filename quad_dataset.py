from torch.utils.data import Dataset
import pandas as pd
from sklearn import preprocessing
from pandas.api.types import is_string_dtype, is_numeric_dtype
import torch
from sklearn import model_selection
import numpy as np


class QuadDataset(Dataset):
    def __init__(self, is_train):
        self.is_train = is_train
        #self.file_location = "out.csv"
        #csv_data = pd.read_csv(self.file_location)
        a = np.random.random(1000).reshape(-1, 1)
        asq = a * a * a
        data = np.concatenate((a, asq), 1)
        csv_data = pd.DataFrame(data, columns=["x", "y"])
        self.total = len(csv_data)
        df = pd.DataFrame(csv_data)

        train, test = model_selection.train_test_split(df, test_size=0.4)
        df = train
        if not self.is_train:
            df = test

        self.x = torch.tensor(df["x"].values, dtype=torch.float32).reshape(-1,1)
        self.y = torch.tensor(df["y"].values, dtype=torch.float32)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

if __name__ == "__main__":
    d = QuadDataset(is_train=True)
    from torch.utils.data import DataLoader
    dl = DataLoader(d, batch_size=20000)
    for x,y in dl:
        print(x.shape)
        print(y)
        print(torch.max(x))
        print(torch.min(x))
        exit(0)


