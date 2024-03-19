import pandas as pd
from mflow.preprocessor import DataFramePreprocessor


if __name__ == "__main__":
    df = pd.read_csv("buggy.csv")
    preprocessor = DataFramePreprocessor(df=df)
    #preprocessor.info()
    #preprocessor.remove_missing_values()
    #print(preprocessor.df.isnull().sum())
    print(preprocessor.handle_all_missing_values())
