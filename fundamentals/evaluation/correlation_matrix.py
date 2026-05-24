import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = {
    'age': [20,25,30,35,40],
    'salary': [30,40,50,60,70],
    'experience': [1,3,5,7,9]
}


df = pd.DataFrame(data)

corr = df.corr()
print(corr)

sns.heatmap(
    corr,
    annot=True,
    cmap='coolwarm'
)

plt.show()

# Used for:

# feature selection
# multicollinearity detection
# exploratory data analysis
# dimensionality reduction