
# ROC and AUC evaluate classification models across thresholds.

# Very important for binary classification.

# Receiver Operating Characteristic
# ROC Curve

# Plots:
# Axis	Metric
# X-axis	False Positive Rate
# Y-axis	True Positive Rate
# AUC Interpretation
# AUC	Meaning
# 1.0	perfect classifier
# 0.5	random guessing
# <0.5	poor classifier


from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt


x,y=make_classification(
    n_samples=1000,
    n_features=10,
    random_state=43
)

x_train,x_test,y_train,y_test = train_test_split(
    x,y,test_size=0.2,random_state=43
)

model = LogisticRegression()
model.fit(x_train,y_train)

y_prob = model.predict_proba(x_test)[:,1]

# roc
fpr , tpr , thresholds = roc_curve(
    y_test,
    y_prob
)


roc_auc = auc(fpr,tpr)

plt.plot(
    fpr,tpr,
    label=f'auc = {roc_auc:.2f}')

plt.plot([0,1],[0,1],'--')

plt.xlabel("false positive rate")
plt.ylabel("truc positive rate")

plt.legend()
plt.show()
# Ideal ROC Curve

# Best classifier hugs:

# top-left corner

# because:

# high TPR
# low FPR