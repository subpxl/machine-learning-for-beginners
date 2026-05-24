from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

x,y = make_classification(
    n_samples=1000,
    n_features=10,
    random_state=42
)

model =LogisticRegression()

scores = cross_val_score(
    model,
    x,
    y,
    cv=5
)

print(scores)
print("mean cv score",scores.mean())