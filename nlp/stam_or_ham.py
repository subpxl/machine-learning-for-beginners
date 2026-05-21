from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# training data
X_train = [
    "Win a free lottery now",
    "Claim your free prize",
    "Meeting at 10 tomorrow",
    "Project report attached",
    "love you babe"
]

y_train = ["spam", "spam", "ham", "ham","spam"]

# pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("clf", MultinomialNB())
])

# train
model.fit(X_train, y_train)

# predict
texts = [
    "free money available",
    "schedule the meeting",
    "love me baby"
]

predictions = model.predict(texts)

print(predictions)