from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
data = load_iris()
X = data.data
y = data.target
model = DecisionTreeClassifier(random_state=42)
scores = cross_val_score(model, X, y, cv=5)
print("Cross Validation Scores:", scores)
print("Average Accuracy:", scores.mean())
