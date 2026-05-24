from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix


x,y =make_classification(
    n_samples=1000,
    n_features=10,
    random_state=32
)

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=43)


model = LogisticRegression()

model.fit(x_train,y_train)

y_pred= model.predict(x_test)

cm = confusion_matrix(y_test,y_pred)
print(cm)

import seaborn as sns
import matplotlib.pyplot as plt

sns.heatmap(
    cm,
    annot=True,
    fmt='d'
)
plt.xlabel('predicted')
plt.ylabel('acutual')
plt.show()