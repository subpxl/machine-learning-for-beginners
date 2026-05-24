# Core Idea

# Prediction error can be decomposed roughly into:

# Error=Bias
# 2
# +Variance+Noise

# Error=Bias
# 2
# +Variance+Noise

# Symptoms of High Bias
# poor training accuracy
# poor testing accuracy
# overly simple model

# Examples

# High-bias models:

# linear regression on complex data
# shallow trees
# overly regularized models

# 3. What is Variance?

# Variance means:

# Model is too sensitive to training data.

# High-variance models:

# memorize training data
# overfit
# fail to generalize

# Symptoms
# excellent training accuracy
# poor testing accuracy

# Examples

# High-variance models:

# deep decision trees
# overly complex neural networks

# 6. Bias-Variance Tradeoff Curve

# As model complexity increases:

# Complexity	Bias	Variance
# low	high	low
# medium	balanced	balanced
# high	low	high

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

# Data
np.random.seed(42)

X = np.linspace(0, 10, 100)
y = np.sin(X) + np.random.normal(0, 0.2, 100)

X = X.reshape(-1,1)

# Polynomial degrees
degrees = [1, 3, 20]

plt.figure(figsize=(12,4))

for i, degree in enumerate(degrees):

    model = make_pipeline(
        PolynomialFeatures(degree),
        LinearRegression()
    )

    model.fit(X, y)

    y_pred = model.predict(X)

    plt.subplot(1,3,i+1)

    plt.scatter(X, y)
    plt.plot(X, y_pred)

    plt.title(f"Degree {degree}")

plt.tight_layout()
plt.show()