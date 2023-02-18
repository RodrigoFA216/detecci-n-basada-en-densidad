import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor

# Generar datos sintéticos
np.random.seed(40) #numpy configura la generación de datos random con una semilla 42
X2=np.random.randint(low=0, high=4, size=100)
Y2=np.random.randint(low=0, high=4, size=100)
X = 0.5 * np.random.randn(100, 2) # genera datos de 100 filas por dos columnas 
XA = np.round(5 * np.random.randn(100, 2))#<--------------------
# el valor que multiplica a np.random genera la dispersión de datos.
#mientras más alto sea el valor más dispersos están los datos
#genera datos que funcionarán de outliers entre -4 y 4 de 20 filas y dos columnas
X_outliers = np.random.uniform(low=0, high=4, size=(20, 2)) 
#X_outliers = np.random.randint(low=0, high=5, size=(20, 2))#<--------------------
#concatenamos en x las matrices que ya habíamos hecho, 
# en primer lugar concatenamos x desplazado en 2 y x desplazado en -2 (en x y y) y concatenamos los outliers 
# esto va a generar una dispersión atípica de los valores outliers
# no importa el orden en que sean concatenados mientras estén en un solo objeto
# print(X2, len(X2), type(X2))
# print(X, len(X), type(X))
# print(X_outliers, len(X_outliers))
X = np.r_[X + 2, X_outliers]

# Ajustar el modelo de detección de outliers
# crea un objeto del modelo de detección basado en densidad llamado LocalOutlierFactor
# un punto es considerado un outlier si su densidad de vecinos es muy baja en comparación 
# con la densidad de los vecinos de los demás puntos
# La variable n_neighbors se utiliza para especificar el número de vecinos más cercanos que 
# se utilizarán para calcular la densidad
clf = LocalOutlierFactor(n_neighbors=50)
# se ajusta el modelo de detección dando como parámetros el conjunto X que 
# recordando el conjunto original se transformó en x+2 y x-2 dentro de x
y_pred = clf.fit_predict(X)

# Graficar los resultados
#se crea un objeto colors donde se etiquetarán de azul 377eb8 los outliers y de 
# naranja ff7f00los datos que son representativos
colors = np.array(['#377eb8', '#ff7f00'])
# Grafica primera columna en x y segunda columna en Y
# usa colors para asignar colores a los puntos y_pred contiene las 
# etiquetas de predicción devueltas por el modelo LocalOutlierFactor
plt.scatter(X[:, 0], X[:, 1], color=colors[(y_pred + 1) // 2])
# La expresión color=colors[(y_pred + 1) // 2] se utiliza para asignar un color a 
# cada punto en función de su etiqueta de predicción. 
# Si un punto es considerado un outlier (y_pred = -1), se le asigna el primer color de 
# la lista colors, y si no es un outlier (y_pred = 1), se le asigna el segundo color. 
# La expresión (y_pred + 1) // 2 se utiliza para mapear los valores de y_pred a los 
# índices de los colores en la lista
plt.xlim((0, 4))
plt.ylim((0, 4))
plt.show()
