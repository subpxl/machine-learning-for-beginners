import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder


NUMERICAL_FEATURES = [
    "tenure",
    "monthly_charges",
    "support_calls"
]

CATEGORICAL_FEATURES = [
    "contract_type",
    "internet_service",
    "payment_method"
]

TARGET_COLUMN = "churn"


def load_data(path):
    df = pd.read_csv(path)

    return df


def preprocess_data(df):
    x = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    return x, y


def build_preprocessor():

    preprocessor = ColumnTransformer([
        (
            "num",
            StandardScaler(),
            NUMERICAL_FEATURES
        ),
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            CATEGORICAL_FEATURES
        )
    ])

    return preprocessor