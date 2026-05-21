from sklearn.decomposition import PCA
# import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

# x = np.array([
#     [1,2,3],
#     [4,5,6],
#     [7,8,9]
# ])

data = load_iris()
x = data.data

pca = PCA(n_components=3)

reduced = pca.fit_transform(x)
print(reduced)

plt.scatter(reduced[:,0],reduced[:,],c="y")

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA Visualization")
plt.show()