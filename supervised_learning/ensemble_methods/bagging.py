# bagging trains many models

from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Generate dataset
X, y = make_classification(
    n_samples=1000,
    n_features=10,
    random_state=42
)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Base model
tree = DecisionTreeClassifier()

# Bagging model
bagging_model = BaggingClassifier(
    estimator=tree,
    n_estimators=50,
    random_state=42
)

# Train
bagging_model.fit(X_train, y_train)

# Predict
y_pred = bagging_model.predict(X_test)

# Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))