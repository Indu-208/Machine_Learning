from sklearn.linear_model import LogisticRegression
X = [[1], [2], [3], [4], [5], [6]]
y = [0, 0, 0, 1, 1, 1]
model = LogisticRegression()
model.fit(X, y)
prediction = model.predict([[4]])
print("Predicted Class:", prediction[0])
