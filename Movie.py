from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
reviews = [
    "This movie is fantastic",
    "I really enjoyed this film",
    "Amazing acting and excellent story",
    "Worst movie ever",
    "The film was boring",
    "Very disappointing experience"
]
labels = [1, 1, 1, 0, 0, 0]
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(reviews)
model = MultinomialNB()
model.fit(X, labels)
new_review = ["The movie was amazing and fantastic"]
result = model.predict(vectorizer.transform(new_review))
print("Review:", new_review[0])
print("Sentiment:", "Positive" if result[0] == 1 else "Negative")
