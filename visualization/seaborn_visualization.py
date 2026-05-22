import seaborn as sns
import matplotlib.pyplot as plt

iris = sns.load_dataset("iris")

print(iris.head())

sns.scatterplot(
    data=iris,
    x="sepal_length",
    y="sepal_width",
        hue="species"

)

plt.show()

sns.histplot(
    data=iris,
    x="petal_length",
        hue="species"
        # kde=True    
)

plt.show()

sns.kdeplot(
    data=iris,
    x="petal_length"
)

plt.show()

sns.boxplot(
    data=iris,
    x="species",
    y="petal_length"
)

plt.show()

sns.violinplot(
    data=iris,
    x="species",
    y="petal_length",
    hue="species"

)

plt.show()

corr = iris.corr(numeric_only=True)

sns.heatmap(
    corr,
    annot=True
)
plt.show()

sns.regplot(
    data=iris,
    x="sepal_length",
    y="petal_length"
)
plt.show()


g = sns.FacetGrid(
    iris,
    col="species"
)

g.map(
    plt.scatter,
    "sepal_length",
    "petal_length"
)

plt.show()