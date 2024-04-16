import pandas as pd
from mflow.regression import SimpleRegression
from mflow.image_classification import PretrainImageClassifier

def test_regression():
    df = pd.read_csv("housing.csv")
    regression = SimpleRegression(df=df, target_col_name="median_house_value")
    regression.fit()



if __name__ == "__main__":
    train_path = "./image_dataset/classification/train"
    val_path = "./image_dataset/classification/val"


    image_classifier = PretrainImageClassifier(
            train_data_path=train_path,
            val_data_path=val_path,
            )


    image_classifier.fit()

