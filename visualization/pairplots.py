import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import pandas as pd
import numpy as np

iris=load_iris()

df = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

df['species']=iris.target

sns.pairplot(
    df,
    hue='species'
)

plt.show()

# Random dataset
df = pd.DataFrame({
    "A": np.random.randn(100),
    "B": np.random.randn(100),
    "C": np.random.randn(100)
})

sns.pairplot(df)

plt.show()