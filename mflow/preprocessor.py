from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
import numpy as np

class DataFramePreprocessor:
    def __init__(self, df):
        self.df = df
        self.original_df = df.copy()

        col_names = self.df.columns
        self.data_types = {}

        for index, dt in enumerate(self.df.dtypes):
            self.data_types[col_names[index]] = dt



    def info(self):

        print("************* Head **************")
        print(self.df.head())
        print("************* Describe **************")
        print(self.df.describe())
        print("************* Info **************")
        print(self.df.info())
        print("************* Unique **************")

        for col_name in self.df.columns:
            unique_data = self.df[col_name].unique()
            print(f"{col_name} (total {len(unique_data)}) => {unique_data}")



        print("************* Null Count **************")
        print(f"{self.df.isnull().sum()}")

    def handle_all_missing_values(self):
        self.handle_missing_value_numbers()
        self.remove_missing_values()
        print(self.df.isnull().sum())



    def remove_missing_values(self, remove_name="row"):
        if remove_name == "row":
            self.df = self.df.dropna(axis=0)
            return

        # column
        self.df = self.df.dropna(axis=1)



    def handle_missing_value_numbers(self, strategy_name="mean"):
        num_col_flags = []

        for type_ in self.data_types.values():
            flag = False
            if type_ != "object":
                flag = True

            num_col_flags.append(flag)

        all_cols = list(self.data_types.keys())
        num_cols = []

        for index, flag in enumerate(num_col_flags):
            if flag == False:
                continue

            num_cols.append(all_cols[index])

        imputer = SimpleImputer(missing_values=np.nan, strategy=strategy_name)
        imputer.fit(self.df[num_cols].values)
        self.df[num_cols] = imputer.transform(self.df[num_cols])


    def encode_text_data(self, strategy_name="label_encoder"):
        text_col_flags = []

        for type_ in self.data_types.values():
            flag = False
            if type_ == "object":
                flag = True

            text_col_flags.append(flag)

        all_cols = list(self.data_types.keys())
        text_cols = []

        for index, flag in enumerate(text_col_flags):
            if flag == False:
                continue

            text_cols.append(all_cols[index])

        print(text_cols)

        if strategy_name == "label_encoder":
            self.label_encoders = {}

            for col_name in text_cols:
                encoder = LabelEncoder()
                self.df[col_name] = encoder.fit_transform(self.df[col_name].values)
                self.label_encoders[col_name] = encoder


            print(self.df.head())






























