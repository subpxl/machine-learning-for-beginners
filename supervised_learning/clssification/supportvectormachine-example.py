from sklearn.svm import SVC
import numpy as np


x = np.array([
    [1],
    [2],
    [3],
    [4]
])


y = np.array([0,0,1,1])

model = SVC()

model.fit(x,y)

prediction = model.predict([[3]])

print("prediction",prediction[0])