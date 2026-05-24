from sklearn.feature_extraction.text import TfidfVectorizer

documents = [
    "machine learning is powerful",
    "deep learning uses neural networks",
    "machine learning and deep learning"
]


vectorizer = TfidfVectorizer()

x = vectorizer.fit_transform(documents)
print(vectorizer.get_feature_names_out())
print(x.toarray())

# Feature extraction means transforming raw data into numerical representations that machine learning models can learn from.

# A few foundational examples are below.

# Text Feature Extraction — TF-IDF

# Raw text cannot be directly used by most ML models. TF-IDF converts text into numerical vectors.


# python .\feature_extraction.py
# ['and' 'deep' 'is' 'learning' 'machine' 'networks' 'neural' 'powerful'
#  'uses']
# [[0.         0.         0.5844829  0.34520502 0.44451431 0.
#   0.         0.5844829  0.        ]
#  [0.         0.38376993 0.         0.29803159 0.         0.50461134
#   0.50461134 0.         0.50461134]
#  [0.53058735 0.40352536 0.         0.62674687 0.40352536 0.
#   0.         0.         0.        ]]



import pandas as pd

df = pd.DataFrame({
    "color":["red","blue","green","red"]
})

encoded = pd.get_dummies(df["color"])
print(encoded)


from sklearn.preprocessing import PolynomialFeatures
import numpy as np

x = np.array([[2],[3],[4]])

poly = PolynomialFeatures(degree=2)
x_poly = poly.fit_transform(x)
print(x_poly)