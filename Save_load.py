import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
data = load_iris()
model = RandomForestClassifier(random_state=42)
model.fit(data.data, data.target)
joblib.dump(model, "iris_model.pkl")
loaded_model = joblib.load("iris_model.pkl")
prediction = loaded_model.predict([[5.1, 3.5, 1.4, 0.2]])
print("Predicted Flower:",
      data.target_names[prediction[0]])
