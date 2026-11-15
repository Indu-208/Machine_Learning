from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
iris = load_iris()
X = iris.data
y = iris.target
model = DecisionTreeClassifier(random_state=42)
model.fit(X, y)
sample = [[6.1, 2.8, 4.7, 1.2]]
prediction = model.predict(sample)
print("Flower Name:", iris.target_names[prediction[0]])
