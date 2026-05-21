from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline


def split_data(x, y):
    return train_test_split(
        x, y, test_size=0.2, random_state=43
    )


def train_model(x_train, y_train):
    num_features = ["area", "bedrooms", "bathrooms", "parking"]
    cat_features = ["location"]

    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, num_features),
        ("cat", categorical_transformer, cat_features)
    ])

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(random_state=43))
    ])

    param_grid = {
        "regressor__n_estimators": [50, 100, 200],
        "regressor__max_depth": [None, 10, 20],
        "regressor__min_samples_split": [2, 5]
    }

    grid = GridSearchCV(
        model,
        param_grid,
        cv=2,
        scoring="r2",
        n_jobs=-1
    )

    grid.fit(x_train, y_train)
    print("best parameters:", grid.best_params_)
    print("training complete")
    return grid.best_estimator_