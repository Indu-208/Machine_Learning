from sklearn.ensemble import RandomForestClassifier
X = [[1, 2], [2, 3], [3, 4],
     [6, 7], [7, 8], [8, 9]]
y = [0, 0, 0, 1, 1, 1]
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)
prediction = model.predict([[7, 7]])
print("Predicted Class:", prediction[0])
