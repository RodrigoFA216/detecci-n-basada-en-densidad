import numpy as np
import matplotlib.pyplot as plt

# Genera el primer conjunto de datos aleatorios
x1 = np.random.normal(0, 1, 100)
y1 = np.random.normal(0, 1, 100)

# Genera el segundo conjunto de datos aleatorios
x2 = np.random.normal(5, 1, 100)
y2 = np.random.normal(5, 1, 100)

# Grafica el primer conjunto de datos
plt.scatter(x1, y1, c='r')

# Grafica el segundo conjunto de datos
plt.scatter(x2, y2, c='b')

# Calcula la media de los valores de x e y de ambos conjuntos
x_medio = (np.mean(x1) + np.mean(x2)) / 2
y_medio = (np.mean(y1) + np.mean(y2)) / 2

# Grafica el punto en el medio
plt.scatter(x_medio, y_medio, c='g', marker='x', s=100)

# Pide al usuario que ingrese el radio de búsqueda
radio = float(input("Ingrese el radio de búsqueda: "))

# Calcula la distancia entre el punto medio y cada punto de la primera dispersión
distancias1 = np.sqrt((x1 - x_medio)**2 + (y1 - y_medio)**2)

# Calcula la distancia entre el punto medio y cada punto de la segunda dispersión
distancias2 = np.sqrt((x2 - x_medio)**2 + (y2 - y_medio)**2)

# Cuenta el número de puntos cercanos al punto medio en cada grupo
puntos_cercanos_1 = np.sum(distancias1 < radio)
puntos_cercanos_2 = np.sum(distancias2 < radio)

# Determina a qué dispersión pertenece el punto medio
if np.min(distancias1) < np.min(distancias2):
    pertenece_a = 'dispersión 1'
else:
    pertenece_a = 'dispersión 2'

# Muestra a qué dispersión pertenece el punto medio y el conteo de puntos cercanos en cada grupo
print('El punto medio pertenece a la', pertenece_a)
print('El punto medio está cerca de', puntos_cercanos_1, 'puntos del grupo 1 y', puntos_cercanos_2, 'puntos del grupo 2')

# Grafica los puntos cercanos al punto medio según el radio ingresado
puntos_cercanos_x1 = x1[np.array(distancias1) < radio]
puntos_cercanos_y1 = y1[np.array(distancias1) < radio]
plt.scatter(puntos_cercanos_x1, puntos_cercanos_y1, c='r')

puntos_cercanos_x2 = x2[np.array(distancias2) < radio]
puntos_cercanos_y2 = y2[np.array(distancias2) < radio]
plt.scatter(puntos_cercanos_x2, puntos_cercanos_y2, c='b')

# Muestra la gráfica resultante
plt.show()
