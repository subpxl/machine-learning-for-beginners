from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------
# Generate Dataset
# ---------------------------------------------------

np.random.seed(42)

# Class 0
class0 = np.random.normal(
    loc=2,
    scale=0.8,
    size=(50, 1)
)

# Class 1
class1 = np.random.normal(
    loc=6,
    scale=0.8,
    size=(50, 1)
)

# Features
X = np.vstack((class0, class1))

# Labels
y = np.array([0] * 50 + [1] * 50)

# ---------------------------------------------------
# Train-Test Split
# ---------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ---------------------------------------------------
# Base Model
# ---------------------------------------------------

model = KNeighborsClassifier()

# ---------------------------------------------------
# Hyperparameter Grid
# ---------------------------------------------------

params = {
    "n_neighbors": [1, 3, 5, 7, 9],
    "weights": ["uniform", "distance"],
    "metric": ["euclidean", "manhattan"]
}

# ---------------------------------------------------
# Grid Search
# ---------------------------------------------------

grid = GridSearchCV(
    estimator=model,
    param_grid=params,
    cv=5,
    scoring="accuracy"
)

grid.fit(X_train, y_train)

# ---------------------------------------------------
# Best Model
# ---------------------------------------------------

best_model = grid.best_estimator_

print("Best Parameters:")
print(grid.best_params_)

print("\nBest Cross Validation Score:")
print(grid.best_score_)

# ---------------------------------------------------
# Predictions
# ---------------------------------------------------

y_pred = best_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nTest Accuracy:")
print(accuracy)

# ---------------------------------------------------
# Predict New Value
# ---------------------------------------------------

sample = [[4.5]]

prediction = best_model.predict(sample)

print("\nPrediction for 4.5:")
print(prediction[0])

# ---------------------------------------------------
# Visualization
# ---------------------------------------------------

x_range = np.linspace(0, 9, 500).reshape(-1, 1)

predictions = best_model.predict(x_range)

plt.figure(figsize=(10, 6))

# Plot training points
plt.scatter(
    X_train[y_train == 0],
    y_train[y_train == 0],
    s=60,
    label="Class 0"
)

plt.scatter(
    X_train[y_train == 1],
    y_train[y_train == 1],
    s=60,
    label="Class 1"
)

# Plot prediction boundary
plt.plot(
    x_range,
    predictions,
    linewidth=2,
    label="Decision Boundary"
)

# Plot sample point
plt.scatter(
    sample,
    [prediction[0]],
    s=200,
    marker="X",
    label="New Sample"
)

plt.xlabel("Feature Value")
plt.ylabel("Class")
plt.title("KNN Classification")

plt.legend()
plt.grid(True)

plt.show()