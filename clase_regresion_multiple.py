import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# cargar dataset
df = pd.read_csv('housing.csv')
print("Dataset Cargado:", df.shape)

# Preparar el dataset
df['total_bedrooms'] = df['total_bedrooms'].fillna(df['total_bedrooms'].median())
print('Verificacion despues de rellenar con la mediana', df.isnull().sum().sum())

# implementando one-hot-encoding 
df_enc = pd.get_dummies(df, columns=['ocean_proximity'], drop_first=False)
print('Columnas despues de encoding', df_enc.shape[1])

# Separar X, y "DIVIDIR" train/test
X = df_enc.drop('median_house_value', axis=1)
y = df_enc['median_house_value']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("X_train:", X_train.shape, "X_test:", X_test.shape)

# Crear el modelo - Regresion Lineal multiple

modelo = LinearRegression()
modelo.fit(X_train, y_train)

print("Coeficientes:", modelo.coef_)
for nombre, coef in zip(X_train.columns, modelo.coef_):
    print(f" {nombre}: {round(coef, 2)}")
print("Intercepto:", round(modelo.intercept_, 2))

# Metricas

y_pred = modelo.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

print("MEA (Error Absoluto Medio):", round(mae, 2))
print("MSE (Error Cuadrático Medio):", round(mse, 2))
print("RMSE (Raíz del Error Cuadrático Medio):", round(rmse, 2))
print("R² (Coeficiente de Determinación):", round(r2, 2))

# comparativa de modelos: Regresion lineal simple vs multiple
modelo_simple = LinearRegression()
modelo_simple.fit(X_train[['median_income']], y_train)
y_pred_simple = modelo_simple.predict(X_test[['median_income']])

# Metricas comparativas
r2_simple = r2_score(y_test, y_pred_simple)
mae_simple = mean_absolute_error(y_test, y_pred_simple)

print("Modelo Simple (1 variable) - r2:", round(r2_simple, 4), "| MAE:", round(mae_simple, 2))
print("Modelo multiple (13 variables) - r2:", round(r2, 4), "| MAE:", round(mae, 2))