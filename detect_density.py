import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor

# Generar datos sintéticos
np.random.seed(42) #numpy configura la generación de datos random con una semilla 42
X = 0.3 * np.random.randn(100, 2) # genera datos de 100 filas por dos columnas
#genera datos que funcionarán de outliers entre -4 y 4 de 20 filas y dos columnas
X_outliers = np.random.uniform(low=-4, high=4, size=(20, 2)) 
#concatenamos en x las matrices que ya habíamos hecho, 
# en primer lugar concatenamos x desplazado en 2 y x desplazado en -2 (en x y y) y concatenamos los outliers 
# esto va a generar una dispersión atípica de los valores outliers
# no importa el orden en que sean concatenados mientras estén en un solo objeto
X = np.r_[X + 2, X - 2, X_outliers]

# Ajustar el modelo de detección de outliers
# crea un objeto del modelo de detección basado en densidad llamado LocalOutlierFactor
# un punto es considerado un outlier si su densidad de vecinos es muy baja en comparación 
# con la densidad de los vecinos de los demás puntos
# La variable n_neighbors se utiliza para especificar el número de vecinos más cercanos que 
# se utilizarán para calcular la densidad
clf = LocalOutlierFactor(n_neighbors=10)
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
plt.xlim((-5, 5))
plt.ylim((-5, 5))
plt.show()
