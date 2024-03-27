import pandas as pd
from mflow.preprocessor import DataFramePreprocessor


if __name__ == "__main__":
    df = pd.read_csv("housing.csv")
    preprocessor = DataFramePreprocessor(df=df)
    preprocessor.handle_all_missing_values(show=False)
    preprocessor.encode_text_data(strategy_name="one")

    X_df, y_df = preprocessor.get_x_y(y_name="median_house_value")

    print(X_df.head())
    print(y_df.head())
