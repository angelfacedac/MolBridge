import os

import pandas as pd
from torch.utils.data import Dataset


class Mydata(Dataset):
    def __init__(self, root_path, kind_path):
        super(Mydata, self).__init__()
        self.root_path = root_path
        self.kind_path = kind_path
        self.path = os.path.join(root_path, kind_path)
        self.df = pd.read_csv(self.path)

    def __getitem__(self, idx):
        return self.df.iloc[idx]

    def __len__(self):
        return len(self.df)