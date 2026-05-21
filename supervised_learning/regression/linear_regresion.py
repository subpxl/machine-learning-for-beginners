from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt



x = np.array([[1],[2],[3],[4],[5]])
y = np.array([2,4,6,8,10])

model = LinearRegression()

model.fit(x,y)

predition = model.predict([[6]])

print("predition",predition)

print("slope",model.coef_[0])
print("intercept",model.intercept_)


# predictions for plottin line
y_pred = model.predict(x)
plt.scatter(x,y)
plt.plot(x,y_pred)

plt.xlabel("X")
plt.ylabel("y")
plt.title("Linear Regression")

plt.show()