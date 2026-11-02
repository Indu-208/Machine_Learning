import numpy as np
from sklearn.linear_model import LinearRegression

# Dataset
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([10, 20, 30, 40, 50])

# Create model
model = LinearRegression()

# Train model
model.fit(X, y)

# Predict
prediction = model.predict([[6]])

print("Predicted Value:", prediction[0])
print("Model Accuracy:", model.score(X, y))
