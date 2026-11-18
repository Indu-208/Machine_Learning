from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
data = load_iris()
X = data.data
model = PCA(n_components=2)
result = model.fit_transform(X)
print("Original Shape:", X.shape)
print("Reduced Shape:", result.shape)
print("Reduced Data:\n", result[:5])
