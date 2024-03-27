from torch.utils.data import Dataset
import torch

class RegressionDataset(Dataset):
    def __init__(self, X_df, y_df):
        super(RegressionDataset, self).__init__()

        self.x_tensor = torch.tensor(X_df.values, dtype=torch.float)
        self.y_tensor = torch.tensor(y_df.values, dtype=torch.float)


    def __getitem__(self, idx):
        return self.x_tensor[idx], self.y_tensor[idx]
