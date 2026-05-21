import joblib
import numpy as np

model = joblib.load("models/house_model.pkl")

sample = np.array([1200,3,2,1,1])

prediction = model.predict(sample)

print("predicted price",prediction[0])