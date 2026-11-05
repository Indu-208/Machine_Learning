from sklearn.cluster import KMeans
X = [[1, 2], [2, 3], [3, 4],
     [8, 8], [9, 9], [10, 10]]
model = KMeans(n_clusters=2, random_state=42)
model.fit(X)
print("Cluster Labels:", model.labels_)
print("Cluster Centers:", model.cluster_centers_)
