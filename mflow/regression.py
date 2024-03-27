from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch

from mflow.preprocessor import DataFramePreprocessor
from tqdm import tqdm


class RegressionDataset(Dataset):
    def __init__(self, X_df, y_df, device):
        super(RegressionDataset, self).__init__()

        self.x_tensor = torch.tensor(X_df.values, dtype=torch.float).to(device)
        self.y_tensor = torch.tensor(y_df.values, dtype=torch.float).to(device)

    def __len__(self):
        return len(self.x_tensor)


    def __getitem__(self, idx):
        return self.x_tensor[idx], self.y_tensor[idx]



class SimpleRegression:
    def __init__(self, df, target_col_name):

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        preprocessor = DataFramePreprocessor(df=df)
        preprocessor.handle_all_missing_values(show=False)
        preprocessor.encode_text_data(strategy_name="one")

        X_df, y_df = preprocessor.get_x_y(y_name=target_col_name)


        X_scaled_df = preprocessor.scale(df=X_df)
        y_df[target_col_name] = y_df[target_col_name] / 100000

        self.x_train, self.x_val, self.y_train, self.y_val =  preprocessor.train_test_split(
                X_df=X_scaled_df,
                y_df=y_df
                )

    def fit(self, epochs=100, batch_size=30, learning_rate=0.001):

        train_dataset = RegressionDataset(
                X_df=self.x_train,
                y_df=self.y_train,
                device=self.device
                )

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        val_dataset = RegressionDataset(
                X_df=self.x_val,
                y_df=self.y_val,
                device=self.device
                )

        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)



        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}")
            for item in tqdm(train_dataloader, "Training "):
                x, y = item







