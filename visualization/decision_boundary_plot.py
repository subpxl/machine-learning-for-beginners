import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


x,y = make_moons(n_samples=300,noise=0.25,random_state=32)

x_train,x_test,y_train,y_test = train_test_split(
    x,y,test_size=0.2,random_state=32
)

scaler = StandardScaler()
x_train =scaler.fit_transform(x_train)
x_test= scaler.transform(x_test)

model = SVC(kernel="rbf",gamma="scale",C=1.0)
model.fit(x_train,y_train)

x_min,x_max=x_train[:,0].min()-1, x_train[:,0].max()+1
y_min,y_max=x_train[:,1].min()-1, x_train[:,1].max()+1

xx,yy=np.meshgrid(
    np.linspace(x_min,x_max,500),
    np.linspace(y_min,y_max,500)
)


# predict over grid
grid_points = np.c_[xx.ravel(),yy.ravel()]
Z = model.predict(grid_points)
Z = Z.reshape(xx.shape)


plt.figure(figsize=(8,6))
plt.contourf(xx,yy,Z,alpha=0.3)

plt.scatter(
    x_train[:,0],
    x_train[:,1],
    c=y_train,
    edgecolor="k"
)

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Decision Boundary Visualization")
plt.show()