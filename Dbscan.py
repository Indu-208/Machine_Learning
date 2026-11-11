from sklearn.cluster import DBSCAN
X = [[1, 2], [2, 2], [2, 3],
     [8, 8], [8, 9], [25, 80]]
model = DBSCAN(eps=2, min_samples=2)
labels = model.fit_predict(X)
print("Cluster Labels:", labels)
