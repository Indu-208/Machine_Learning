import numpy as np
from sklearn.ensemble import RandomForestRegressor
X = np.array([
    [1.0, 70],
    [1.2, 85],
    [1.5, 100],
    [1.8, 120],
    [2.0, 140],
    [2.5, 180],
    [3.0, 220]
])
mileage = np.array([22, 20, 18, 16, 14, 11, 9])
model = RandomForestRegressor(random_state=42)
model.fit(X, mileage)
prediction = model.predict([[1.6, 110]])
print("Predicted Mileage:", round(prediction[0], 2), "km/l")
