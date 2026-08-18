import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, roc_auc_score

os.makedirs('graficas', exist_ok=True)

X_train = pd.read_csv('X_train.csv')
X_test = pd.read_csv('X_test.csv')
y_train = pd.read_csv('y_train.csv').squeeze()  # Convertir a Series
y_test = pd.read_csv('y_test.csv').squeeze()  # Convertir a Series

modelo = LogisticRegression(max_iter=1000)
modelo.fit(X_train, y_train)

#prediccion - Probabilidad
y_pred = modelo.predict(X_test)
y_proba = modelo.predict_proba(X_test)[:, 1]  # Probabilidad de la clase positiva

print("Modelo listo. Predicciones generadas:", len(y_test), "Pasajeros de Prueba" )

# Accuracy - prediccion que siempre salga "No sobrevivieron"
prediccion_tonta = [0] * len(y_test)  # Predicción tonta: todos sobreviven

accuracy_tonta = accuracy_score(y_test, prediccion_tonta)
accuracy_modelo = accuracy_score(y_test, y_pred)

print(f"Accuracy de un modelo que siempre predice 'No sobrevivio': {accuracy_tonta:.4f}")
print(f"Accuracy del modelo de regresion logistica: {accuracy_modelo:.4f}")

#Matriz de confusion
matriz = confusion_matrix(y_test, y_pred)
matriz_confusion = confusion_matrix(y_test, y_pred)
print("Matriz de Confusión:")
print(matriz)
print()
print(f"Verdero Negativo (VN): {matriz_confusion[0][0]}, predijo 'no sobrevivio' y acerto")
print(f"Falso Positivo (FP): {matriz_confusion[0][1]}, predijo 'sobrevivio' pero no sobrevivio")
print(f"Falso Negativo (FN): {matriz_confusion[1][0]}, predijo 'no sobrevivio' pero sobrevivio")
print(f"Verdero Positivo (VP): {matriz_confusion[1][1]}, predijo 'sobrevivio' y acerto")


#visualizacion de la matriz de confusion
plt.figure(figsize=(6, 5))
sns.heatmap(matriz_confusion, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Sobrevivio', 'Sobrevivio'],
            yticklabels=['No Sobrevivio', 'Sobrevivio'])
plt.title('Matriz de Confusión')
plt.xlabel('Predicción de modelo')
plt.ylabel('Valor Real')
plt.tight_layout()
plt.savefig('graficas/matriz_confusion.png')
plt.show()

#Formulas

#precision = VP / (VP + FP) 

#recall = VP / (VP + FN) 

#F1 = 2 * (precision * recall) / (precision + recall)


# Metricas Precision, Recall, F1-Score
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")

#Metrica classification report
print(classification_report(y_test, y_pred, target_names=['No Sobrevivio', 'Sobrevivio']))

#Curva ROC y AUC
fpr, tpr, umbrales = roc_curve(y_test, y_proba)
auc = roc_auc_score(y_test, y_proba)

plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, color='steelblue', label=f'Modelo (AUC = {auc:.3f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--', label='Modelo Aleatorio (AUC = 0.5)')
plt.xlabel('Tasa de Falsos Positivos (FPR)')
plt.ylabel('Tasa de Verdaderos Positivos (TPR)')
plt.legend()
plt.tight_layout()
plt.savefig('graficas/03_curva_roc.png')
plt.show()

print(f"AUC (Area bajo la curva ROC): {auc:.4f}")