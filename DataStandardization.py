from sklearn.preprocessing import StandardScaler
X = [[10, 200],
     [20, 300],
     [30, 400],
     [40, 500]]
scaler = StandardScaler()
result = scaler.fit_transform(X)
print("Standardized Data:\n", result)
