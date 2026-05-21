from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt 

x= np.array([
      # Cluster 1
    [1,2],
    [2,1],
    [2,2],
    [3,1],
    [3,2],
    [2,3],

    # Cluster 2
    [8,8],
    [9,8],
    [8,9],
    [9,9],
    [10,8],
    [10,9],

    # Cluster 3
    [15,2],
    [16,1],
    [16,3],
    [17,2],
    [18,1],
    [18,3]
])

model = KMeans(n_clusters=2,random_state=0)

model.fit(x)

print("cluster labels:")
print(model.labels_)


plt.scatter(
            x[:,0],
            x[:,1],
            c=model.labels_,
            s=80
            )

plt.scatter(
    model.cluster_centers_[:,0],
    model.cluster_centers_[:,1],
    marker="X",
    s=300
)

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("K-means clustering")
plt.grid(True)

plt.show()