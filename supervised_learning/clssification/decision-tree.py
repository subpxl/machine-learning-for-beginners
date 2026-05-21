from sklearn.tree import DecisionTreeClassifier,plot_tree
import matplotlib.pyplot as plt
import numpy as np

x = np.array([
    [1],
    [2],
    [3],
    [4]
])


y = np.array([0,0,1,1])

model = DecisionTreeClassifier()
model.fit(x,y)

prediction = model.predict([[3]])

print("prediction",prediction[0])

plt.figure(figsize=(6,4))
plot_tree(model,filled=True)
plt.title("Decision Tree")
plt.show()