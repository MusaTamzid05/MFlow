

class DataFramePreprocessor:
    def __init__(self, df):
        self.df = df
        self.original_df = df.copy()


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


    def handle_missing_value_str(self, remove_name="row"):
        if remove_name == "row":
            self.df = self.df.dropna(axis=0)
            return

        # column
        self.df = self.df.dropna(axis=1)


