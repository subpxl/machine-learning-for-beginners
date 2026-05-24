# Machine learning models cannot understand raw text directly.
# Text must become numerical vectors.
# This process is called vectorization.
# We convert text into numbers.

# Method 1 — Bag of Words

from sklearn.feature_extraction.text import CountVectorizer

documents =[
        "machine learning is fun",
    "deep learning is powerful",
    "machine learning"
]

vectorizer  = CountVectorizer()

x = vectorizer.fit_transform(documents)

print(vectorizer.get_feature_names_out())
print(x.toarray())


# Method 2 — TF-IDF Vectorization
# Problem with Bag of Words:
# common words dominate
# important rare words get ignored
# TF-IDF gives importance weights.

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()

x_tfidf = vectorizer.fit_transform(documents)

print(vectorizer.get_feature_names_out())
print(x_tfidf.toarray())