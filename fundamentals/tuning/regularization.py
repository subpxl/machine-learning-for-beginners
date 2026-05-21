from sklearn.linear_model import Ridge
import numpy as np

x = np.array([[1],[2],[3],[4]])
y = np.array([2,4,6,8])

model = Ridge(alpha=1.0)

model.fit(x,y)
print(model.predict([[5]]))

# lasso regression
from sklearn.linear_model import Lasso

model = Lasso(alpha=0.1)
model.fit(x,y)
print(model.predict([[5]]))