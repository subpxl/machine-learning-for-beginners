import os

from preprocess import load_data, preprocess_data, build_preprocessor
from train import build_model, split_data, train_model, save_model
from evaluate import evaluate_model


def main():
    # Resolve paths relative to project root (one level up from src/)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "data", "churn.csv")
    model_path = os.path.join(base_dir, "models", "churn_model.pkl")

    # Load and split data
    df = load_data(data_path)
    x, y = preprocess_data(df)
    x_train, x_test, y_train, y_test = split_data(x, y)

    # Build pipeline (preprocessor + classifier) and train
    preprocessor = build_preprocessor()
    model = build_model(preprocessor)
    model = train_model(model, x_train, y_train)

    # Evaluate and save
    evaluate_model(model, x_test, y_test)
    save_model(model, model_path)

    print(f"\nModel saved to {model_path}")


if __name__ == "__main__":
    main()