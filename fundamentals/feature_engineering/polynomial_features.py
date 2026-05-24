# Linear models can only learn straight-line relationships.
# But many real relationships are nonlinear:
# Polynomial features create extra columns 

import numpy as np 
from sklearn.preprocessing import PolynomialFeatures

x = np.array([
    [2],
    [3],
    [4]
])

poly = PolynomialFeatures(degree=2)

x_poly = poly.fit_transform(x)
print(x_poly)