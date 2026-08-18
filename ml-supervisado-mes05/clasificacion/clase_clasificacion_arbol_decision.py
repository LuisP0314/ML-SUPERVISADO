import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

X_train = pd.read_csv('X_train.csv')
X_test = pd.read_csv('X_test.csv')
y_train = pd.read_csv('y_train.csv').squeeze()
y_test = pd.read_csv('y_test.csv').squeeze()  # Convertir a Series

print("Datos cargados:", X_train.shape, X_test.shape)

# Modelo de decision
arbol = DecisionTreeClassifier(max_depth=4, random_state=42)
arbol.fit(X_train, y_train)

print("Arbol entrenado. Profundidad maxima configurada", 4)

# comparar los modelos de clasificacion
logistica = LogisticRegression(max_iter=1000)
logistica.fit(X_train, y_train)

acc_logistica = accuracy_score(y_test, logistica.predict(X_test))
acc_arbol = accuracy_score(y_test, arbol.predict(X_test))

print(f"Accuracy regresion logistica - accuracy: {acc_logistica:.4f}")
print(f"Accuracy arbol de decision - accuracy: {acc_arbol:.4f}")

#importancias de las variables
importancias = pd.Series(arbol.feature_importances_, index=X_train.columns)
importancias = importancias.sort_values(ascending=False)

print("Las 5 variables mas importantes para el arbol de decision son:")
print(importancias.head(5).round(3))