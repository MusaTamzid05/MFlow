from torch.utils.data import Dataset
import torch

from mflow.preprocessor import DataFramePreprocessor

class RegressionDataset(Dataset):
    def __init__(self, X_df, y_df):
        super(RegressionDataset, self).__init__()

        self.x_tensor = torch.tensor(X_df.values, dtype=torch.float)
        self.y_tensor = torch.tensor(y_df.values, dtype=torch.float)


    def __getitem__(self, idx):
        return self.x_tensor[idx], self.y_tensor[idx]



class SimpleRegression:
    def __init__(self, df, target_col_name):
        preprocessor = DataFramePreprocessor(df=df)
        preprocessor.handle_all_missing_values(show=False)
        preprocessor.encode_text_data(strategy_name="one")

        X_df, y_df = preprocessor.get_x_y(y_name=target_col_name)


        X_scaled_df = preprocessor.scale(df=X_df)
        y_df[target_col_name] = y_df[target_col_name] / 100000

        x_train, x_val, y_train, y_val =  preprocessor.train_test_split(
                X_df=X_scaled_df,
                y_df=y_df
                )

        print(x_train.shape, y_train.shape, x_val.shape, y_val.shape)

        dataset = RegressionDataset(X_df=X_scaled_df, y_df=y_df)


