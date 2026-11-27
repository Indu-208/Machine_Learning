from sklearn.ensemble import RandomForestClassifier
X = [
    [1000, 2000, 16],
    [2000, 2500, 32],
    [3000, 3000, 64],
    [4000, 3500, 64],
    [6000, 4500, 128],
    [8000, 5000, 256],
    [12000, 5500, 512]
]
y = [0, 0, 1, 1, 2, 2, 2]
model = RandomForestClassifier(random_state=42)
model.fit(X, y)
prediction = model.predict([[8000, 5000, 128]])
labels = ["Low Price", "Medium Price", "High Price"]
print("Predicted Price Range:", labels[prediction[0]])
