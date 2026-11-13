from sklearn.cluster import AgglomerativeClustering
X = [[1, 2], [2, 3], [3, 4],
     [8, 8], [9, 9], [10, 10]]
model = AgglomerativeClustering(n_clusters=2)
labels = model.fit_predict(X)
print("Cluster Labels:", labels)
