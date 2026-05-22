import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde


data = np.random.normal(loc=50,scale=10,size=1000)

plt.hist(data,bins=30)



plt.xlabel("value")
plt.ylabel("frequency")
plt.title("histogram")
plt.show()


kde = gaussian_kde(data)
x = np.linspace(min(data),max(data),500)

plt.plot(x,kde(x))
plt.title("KDE Distribution Plot")
plt.xlabel("Value")
plt.ylabel("Density")
plt.show()

plt.boxplot(data)

plt.title("Box Plot")
plt.show()


import pandas as pd

# Example dataset
df = pd.DataFrame({
    "Age": [21, 25, 30, 35, 40, 45, 50]
})

plt.hist(df["Age"], bins=5)

plt.xlabel("Age")
plt.ylabel("Count")
plt.title("Age Distribution")
plt.show()

fig, ax = plt.subplots()

ax.plot([1,2,3], [4,5,6])

plt.show()