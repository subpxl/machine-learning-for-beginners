import pandas as pd


def load_data(path):
    df = pd.read_csv(path)
    return df


def preprocess_data(df):
    df = df.fillna(df.mean(numeric_only=True))

    x = df.drop("price", axis=1)
    y = df["price"]

    return x, y