import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

os.makedirs('graficas', exist_ok=True)

df = pd.read_csv('housing.csv')

print("Forma del dataset:", df.shape)
print('\nColumnas disponibles:')
print(list(df.columns))
print('\nPrimeras filas:')
print(df.head())
print("\nTipos de datos:") 
print(df.dtypes)

print(df.isnull().sum())
print()
print(df['ocean_proximity'].value_counts())

#modelo regresion lineal simple
X_simple = df[['median_income']]
y = df['median_house_value']

modelo_simple = LinearRegression()
modelo_simple.fit(X_simple, y)

print('Pendientes:', round(modelo_simple.coef_[0], 2))
print('Intercepto:', round(modelo_simple.intercept_, 2))

ingreso_nuevo = pd.DataFrame([[8.0]], columns=['median_income'])
prediccion = modelo_simple.predict(ingreso_nuevo)

print(f'para un ingreso medio de 8.0, el modelo predice: {prediccion[0]:.2f}')

#Visualizacion
plt.figure(figsize=(9, 6))
plt.scatter(df['median_income'], df['median_house_value'], alpha=0.1, color='steelblue')
plt.plot(df['median_income'], modelo_simple.predict(X_simple), color='red', linewidth=2)
plt.title('Regresión lineal simple: Ingreso medio vs Valor de la vivienda')
plt.xlabel('Ingreso medio')
plt.ylabel('Valor medio de la vivienda')
plt.tight_layout()
plt.savefig('graficas/01_regresion_simple.png')
plt.show()