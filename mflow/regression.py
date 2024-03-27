from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
from torch import nn
from torch import optim

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


class SimpleRegressionModel(nn.Module):
    def __init__(self, input_size):
        super(SimpleRegressionModel, self).__init__()

        self.seq = nn.Sequential(
                nn.Linear(input_size, 20),
                nn.ReLU(),

                nn.Linear(20, 10),
                nn.ReLU(),

                nn.Linear(10, 5),
                nn.ReLU(),

                nn.Linear(5, 1),

                )

    def forward(self, x):
        return self.seq(x)



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

        self.model = SimpleRegressionModel(input_size=self.x_train.shape[1])
        self.model = self.model.to(self.device)

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

        loss_fn = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)



        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}")

            self.model.train()

            train_lossses = []

            for item in tqdm(train_dataloader, "Training "):
                x, y = item
                preds = self.model(x)

                loss = loss_fn(y, preds)

                loss.backward()
                optimizer.step()

                train_lossses.append(loss.item())

                optimizer.zero_grad()

            print(f"\tLoss {sum(train_lossses) / len(train_lossses)}")


            self.model.eval()

            val_lossses = []

            for item in tqdm(val_dataloader, "Validation "):
                x, y = item
                preds = self.model(x)

                loss = loss_fn(y, preds)

                val_lossses.append(loss.item())

            print(f"\tLoss {sum(val_lossses) / len(val_lossses)}")







