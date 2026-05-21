from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LinearRegression
import numpy as np

x = np.array([
    [1],
    [2],
    [3],
    [4],
    [5]
])

y = np.array([2,4,6,8,10])

model = LinearRegression()

scores = cross_val_predict(model,x,y,cv=3)

print(scores)