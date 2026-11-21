import numpy as np
from sklearn.cluster import KMeans
customers = np.array([
    [20, 2000],
    [22, 2500],
    [25, 3000],
    [45, 15000],
    [48, 18000],
    [50, 20000],
    [30, 7000],
    [32, 7500]
])
model = KMeans(n_clusters=3, random_state=42, n_init=10)
labels = model.fit_predict(customers)
for i, label in enumerate(labels):
    print("Customer", i + 1, "belongs to Group", label)
print("\nCluster Centers:")
print(model.cluster_centers_)
