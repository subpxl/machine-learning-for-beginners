from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

model = joblib.load("models/house_model.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    features = pd.DataFrame([{
        "area": data["area"],
        "bedrooms": data["bedrooms"],
        "bathrooms": data["bathrooms"],
        "parking": data["parking"],
        "location": data["location"]
    }])

    prediction = model.predict(features)

    return jsonify({
        "predicted_price": float(prediction[0])
    })


if __name__ == "__main__":
    app.run(debug=True)