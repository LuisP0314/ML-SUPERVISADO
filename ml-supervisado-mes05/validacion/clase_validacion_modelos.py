import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

X_train = pd.read_csv('X_train.csv')
X_test = pd.read_csv('X_test.csv')
y_train = pd.read_csv('y_train.csv').squeeze()  # Convertir a Series
y_test = pd.read_csv('y_test.csv').squeeze()  # Convertir a Series

print('Datos listos:', X_train.shape, X_test.shape)

# Arbol profundo

arbol_profundo = DecisionTreeClassifier(max_depth=None, random_state=42)
arbol_profundo.fit(X_train, y_train)

acc_train_profundo = accuracy_score(y_train, arbol_profundo.predict(X_train))
acc_test_profundo = accuracy_score(y_test, arbol_profundo.predict(X_test))

print('Arbol sin limite de profundidad')
print(f'Accuracy en train: {acc_train_profundo:.4f}')
print(f'Accuracy en test: {acc_test_profundo:.4f}')
print(f'Diferencia(Brecha): {acc_train_profundo - acc_test_profundo:.4f}')

#Arbol simple
arbol_simple = DecisionTreeClassifier(max_depth=4, random_state=42)
arbol_simple.fit(X_train, y_train)

acc_train_simple = accuracy_score(y_train, arbol_simple.predict(X_train))
acc_test_simple = accuracy_score(y_test, arbol_simple.predict(X_test))

print('Arbol con limite de profundidad 4')
print(f'Accuracy en train: {acc_train_simple:.4f}')
print(f'Accuracy en test: {acc_test_simple:.4f}')
print(f'Diferencia(Brecha): {acc_train_simple - acc_test_simple:.4f}')

# Underfitting
arbol_underfitting = DecisionTreeClassifier(max_depth=1, random_state=42)
arbol_underfitting.fit(X_train, y_train)

acc_train_underfitting = accuracy_score(y_train, arbol_underfitting.predict(X_train))
acc_test_underfitting = accuracy_score(y_test, arbol_underfitting.predict(X_test))

print('Arbol con limite de profundidad 1 (Una sola pregunta)')
print(f'Accuracy en train: {acc_train_underfitting:.4f}')
print(f'Accuracy en test: {acc_test_underfitting:.4f}')


# Cross-validation
modelo_cv = DecisionTreeClassifier(max_depth=4, random_state=42)
resultados_cv = cross_val_score(modelo_cv, X_train, y_train, cv=5)

print('Resultados de cada uno de los 5 folds):')
print(resultados_cv.round(4))
print()
print(f"Media: {resultados_cv.mean():.4f}")  
print(f"Desviacion estandar: {resultados_cv.std():.4f}")


# Ejercicio de comparacion de valores max_depth
for profundidad in [2, 4, 6, 8, 10, None]:
    modelo_prueba = DecisionTreeClassifier(max_depth=profundidad, random_state=42)
    modelo_prueba.fit(X_train, y_train)
    acc_train = accuracy_score(y_train, modelo_prueba.predict(X_train))
    acc_test = accuracy_score(y_test, modelo_prueba.predict(X_test))
    
    print(f"max)depth={str(profundidad):>4} | train ={acc_train:.4f} | Test={acc_test:.4f} | Brecha={acc_train - acc_test:.4f}")