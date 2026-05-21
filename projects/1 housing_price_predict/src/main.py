from preprocess import load_data
from preprocess import preprocess_data

from train import split_data
from train import train_model
from evaluate import evaluate_model
from utils import save_model

df = load_data("data/housing.csv")

x, y = preprocess_data(df)

x_train, x_test, y_train, y_test = split_data(x, y)

model = train_model(x_train, y_train)

evaluate_model(model, x_test, y_test)

save_model(model, "models/house_model.pkl")