import pandas as pd
from mflow.preprocessor import DataFramePreprocessor


if __name__ == "__main__":
    df = pd.read_csv("buggy.csv")
    preprocessor = DataFramePreprocessor(df=df)
    preprocessor.handle_all_missing_values(show=False)
    preprocessor.encode_text_data(strategy_name="one")
    print(preprocessor.df.head())
