import os
import joblib
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV


def build_model(preprocessor):
    model = Pipeline([
        ("preprocessor", preprocessor),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                random_state=42,
            )
        )
    ])

    return model


def split_data(x, y):
    return train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )


def train_model(model, x_train, y_train):

    model.fit(x_train, y_train)

    return model


def model_tune(model):

    params = {
        "classifier__n_estimators": [100, 200],
        "classifier__max_depth": [5, 10],
        "classifier__min_samples_split": [2, 5]
    }

    grid = GridSearchCV(
        model,
        params,
        cv=5,
        scoring="f1",
        n_jobs=-1
    )

    return grid


def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
