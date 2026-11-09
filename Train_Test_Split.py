from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iriss
data = load_iris()
X = data.data
y = data.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print("Training Samples:", len(X_train))
print("Testing Samples:", len(X_test))
