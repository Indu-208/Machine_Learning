from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
messages = [
    "Win a free mobile phone now",
    "Congratulations you won a lottery",
    "Claim your free prize today",
    "Are you coming to college",
    "Please submit your assignment",
    "Let's meet tomorrow"
]
labels = [1, 1, 1, 0, 0, 0]
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(messages)
model = MultinomialNB()
model.fit(X, labels)
test = ["Congratulations claim your free prize"]
test_data = vectorizer.transform(test)
result = model.predict(test_data)
print("Message:", test[0])
print("Prediction:", "Spam" if result[0] == 1 else "Not Spam")
