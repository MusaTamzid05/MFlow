import pandas as pd
from mflow.preprocessor import DataFramePreprocessor


if __name__ == "__main__":
    df = pd.read_csv("onlinefoods.csv")
    preprocessor = DataFramePreprocessor(df=df)
    preprocessor.info()
