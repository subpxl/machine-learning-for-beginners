from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import numpy as np



x = np.array([
    [22],
    [25],
    [47],
    [52]
])

y = np.array([0,0,1,1])


model = LogisticRegression()

plt.plot(x,y)
plt.xlabel("X")
plt.ylabel("Probability")
plt.title("Sigmoid Curve")
# train
model.fit(x,y)

# predict
result =model.predict([[49]])
print("prediction",result[0])
plt.show()