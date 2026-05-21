import os
from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Resolve model path relative to this file (app/ -> project root -> models/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "churn_model.pkl")

REQUIRED_FIELDS = [
    "tenure",
    "monthly_charges",
    "support_calls",
    "contract_type",
    "internet_service",
    "payment_method"
]

model = None


def load_model():
    global model
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found at {MODEL_PATH}. "
            "Please train the model first by running src/main.py."
        )
    model = joblib.load(MODEL_PATH)


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 503

    data = request.json
    if data is None:
        return jsonify({"error": "Request body must be JSON"}), 400

    # Validate required fields
    missing = [f for f in REQUIRED_FIELDS if f not in data]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    try:
        df = pd.DataFrame([data])

        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)[0][1]

        return jsonify({
            "prediction": int(prediction),
            "churn_probability": round(float(probability), 4)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    load_model()
    app.run(debug=True)