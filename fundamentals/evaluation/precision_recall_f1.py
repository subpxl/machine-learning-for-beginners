# Precision Measures correctness of positive predictions.
# Precision=  TP/FP+TP
# When model predicts positive,
# how often is it correct?


# Recall Measures ability to detect positives.
# Recall=TP/FN+ TP
# How many real positives
# did the model find?

# F1 Score Balances precision and recall.
# F1=2⋅Precision+Recall/Precision⋅Recall​

# When Each Metric Matters
# Metric	Important When
# Precision	false positives costly
# Recall	false negatives costly
# F1	balance needed

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (precision_score,recall_score,f1_score)

x,y = make_classification(
    n_samples=1000,
    n_features=10,
    random_state=32
)

x_train,x_test,y_train,y_test = train_test_split(
    x,y,
    train_size=0.2,
    random_state=43
)

model = LogisticRegression()

model.fit(x_train,y_train)

y_pred = model.predict(x_test)

print("precisoin", precision_score(y_test,y_pred))
print("recall", recall_score(y_test,y_pred))
print("f1", f1_score(y_test,y_pred))