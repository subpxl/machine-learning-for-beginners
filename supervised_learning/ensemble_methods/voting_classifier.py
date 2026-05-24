from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
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


# Base models
lr = LogisticRegression()
dt = DecisionTreeClassifier()
svm = SVC(probability=True)

# Voting classifier
voting = VotingClassifier(
    estimators=[
        ('lr', lr),
        ('dt', dt),
        ('svm', svm)
    ],
    voting='soft'
)

# Train
voting.fit(X_train, y_train)

# Predict
y_pred = voting.predict(X_test)

# Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))