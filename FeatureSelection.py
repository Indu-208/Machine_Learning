import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.feature_selection import SelectKBest, f_classif
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target
selector = SelectKBest(score_func=f_classif, k=5)
X_selected = selector.fit_transform(X, y)
selected_features = X.columns[selector.get_support()]
print("Original Number of Features:", X.shape[1])
print("Selected Number of Features:", X_selected.shape[1])
print("\nSelected Features:")
for feature in selected_features:
    print(feature)
print("\nSelected Feature Data:")
print(X_selected[:5])
