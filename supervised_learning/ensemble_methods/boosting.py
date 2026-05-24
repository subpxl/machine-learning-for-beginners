# Boosting is an ensemble learning method where models are trained sequentially, 
# and each new model focuses on correcting the mistakes of previous models.
# AdaBoost
# Gradient Boosting
# XGBoost
# LightGBM
# CatBoost

# Boosting builds models like this:

# Model 1 → mistakes
#         ↓
# Model 2 focuses on mistakes
#         ↓
# Model 3 focuses on remaining mistakes

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Dataset
X, y = make_classification(
    n_samples=1000,
    n_features=10,
    random_state=42
)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Weak learner
tree = DecisionTreeClassifier(max_depth=1)

# AdaBoost
model = AdaBoostClassifier(
    estimator=tree,
    n_estimators=50,
    learning_rate=1.0,
    random_state=42
)

# Train
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
print(accuracy_score(y_test, y_pred))