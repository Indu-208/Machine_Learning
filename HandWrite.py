from sklearn.datasets import load_digits
from sklearn.neighbors import KNeighborsClassifier
data = load_digits()
X = data.data
y = data.target
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X, y)
sample = [X[100]]
prediction = model.predict(sample)
print("Actual Digit:", y[100])
print("Recognized Digit:", prediction[0])
