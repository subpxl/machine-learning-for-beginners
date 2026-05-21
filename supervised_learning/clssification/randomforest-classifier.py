from sklearn.ensemble import RandomForestClassifier
import numpy as np

x = np.array([
    [1],
    [2],
    [3],
    [4]
])

y = np.array([0,0,1,1])

model = RandomForestClassifier(n_estimators=10)

model.fit(x,y)

prediction = model.predict([[5]])

print("predictoin",prediction[0])