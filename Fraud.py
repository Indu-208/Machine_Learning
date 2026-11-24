from sklearn.ensemble import IsolationForest
import numpy as np
transactions = np.array([
    [100, 1], [150, 2], [200, 1],
    [250, 3], [180, 2], [300, 1],
    [120, 1], [5000, 10], [8000, 15]
])
model = IsolationForest(
    contamination=0.2,
    random_state=42
)
model.fit(transactions)
results = model.predict(transactions)
for transaction, result in zip(transactions, results):
    print(
        "Amount:", transaction[0],
        "| Status:", "Suspicious" if result == -1 else "Normal"
    )
