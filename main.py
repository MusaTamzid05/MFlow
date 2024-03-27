import pandas as pd
from mflow.preprocessor import DataFramePreprocessor
from mflow.regression import RegressionDataset
from mflow.regression import SimpleRegression

def test():
    df = pd.read_csv("housing.csv")
    preprocessor = DataFramePreprocessor(df=df)
    preprocessor.handle_all_missing_values(show=False)
    preprocessor.encode_text_data(strategy_name="one")

    X_df, y_df = preprocessor.get_x_y(y_name="median_house_value")


    X_scaled_df = preprocessor.scale(df=X_df)
    y_df["median_house_value"] = y_df["median_house_value"] / 100000

    dataset = RegressionDataset(X_df=X_scaled_df, y_df=y_df)

    for item in dataset:
        x, y = item
        print(x.shape, y.shape)





if __name__ == "__main__":
    df = pd.read_csv("housing.csv")
    regression = SimpleRegression(df=df, target_col_name="median_house_value")

