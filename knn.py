from sklearn.neighbors import KNeighborsClassifier
X = [[1, 2], [2, 3], [3, 4],
     [6, 7], [7, 8], [8, 9]]
y = [0, 0, 0, 1, 1, 1]
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X, y)
result = model.predict([[2, 4]])
print("Predicted Class:", result[0])
