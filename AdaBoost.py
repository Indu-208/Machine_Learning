from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import load_iris
data = load_iris()
X = data.data
y = data.target
model = AdaBoostClassifier(
    n_estimators=50,
    random_state=42
)
model.fit(X, y)
prediction = model.predict([[5.1, 3.5, 1.4, 0.2]])
print("Predicted Flower:", data.target_names[prediction[0]])
