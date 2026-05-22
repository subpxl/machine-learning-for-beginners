import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import numpy as np

# Create dataset
X, y = make_blobs(
    n_samples=300,
    centers=4,
    random_state=42
)

# Plot features
plt.scatter(
    X[:,0],
    X[:,1],
    c=y
)

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Feature Visualization")

plt.show()

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
model.fit(X, y)
importance = model.feature_importances_

plt.bar(
    range(len(importance)),
    importance
)
plt.xlabel("Feature Index")
plt.ylabel("Importance")

plt.title("Feature Importance")

plt.show()

from sklearn.decomposition import PCA
pca = PCA(n_components=2)

X_pca = pca.fit_transform(X)

plt.scatter(
    X_pca[:,0],
    X_pca[:,1],
    c=y
)

plt.title("PCA Feature Visualization")

plt.show()