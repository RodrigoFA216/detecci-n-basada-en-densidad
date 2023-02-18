#Random Forest es un algoritmo de aprendizaje automático basado en árboles de decisión que utiliza el ensamblaje de múltiples árboles para mejorar la precisión de las predicciones.
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

# Generar datos sintéticos
X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, n_redundant=0, random_state=42)

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Entrenamiento de modelo Random Forest
rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
rf.fit(X_train, y_train)

# Predicciones
y_pred = rf.predict(X_test)

# Evaluación del modelo
print("Exactitud: ", accuracy_score(y_test, y_pred))
print("Reporte de clasificación: \n", classification_report(y_test, y_pred))
