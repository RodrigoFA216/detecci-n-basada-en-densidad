import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# Generar datos sintéticos
np.random.seed(42)
X = 0.3 * np.random.randn(100, 2)
X_outliers = np.random.uniform(low=-4, high=4, size=(20, 2))
X = np.r_[X + 2, X - 2, X_outliers]
y = np.concatenate([np.ones(100), np.zeros(20)])

# Ajustar el modelo de detección de outliers
clf = LocalOutlierFactor(n_neighbors=20)
y_pred = clf.fit_predict(X)
y_pred = y_pred[:120]

# Separar los datos en outliers y no outliers
mask = np.where(y_pred == 1)[0]
X_train, X_test, y_train, y_test = train_test_split(X[mask], y[mask], test_size=0.33, random_state=42)

# Ajustar un árbol de decisión en los datos no outliers
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)

# Evaluar el rendimiento del árbol de decisión
print("Precisión en los datos de entrenamiento:", dt.score(X_train, y_train))
print("Precisión en los datos de prueba:", dt.score(X_test, y_test))

