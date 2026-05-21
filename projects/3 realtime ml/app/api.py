from fastapi import FastAPI
from pydantic import BaseModel
import logging
from pydantic import Field

import pandas as pd
import joblib

app = FastAPI()


from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "fraud_model.pkl"

model = joblib.load(MODEL_PATH)
model = joblib.load("models/fraud_model.pkl")


class Transaction(BaseModel):
    Time: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float  = Field(gt=0)



@app.post("/predict")
async def predict(data:Transaction):
    input_data = pd.DataFrame([data.dict()])
    prediction = model.predict(input_data)[0]

    probability=model.predict_proba(input_data)[0][1]

    return {
        "prediction":int(prediction),
        "fraud_probability":float(probability)
    }

