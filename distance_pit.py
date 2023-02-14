import matplotlib.pyplot as plt
import numpy as np
import math

# Genera el primer conjunto de datos aleatorios
x1 = np.random.normal(0, 1, 100)
y1 = np.random.normal(0, 1, 100)

# Genera el segundo conjunto de datos aleatorios
x2 = np.random.normal(5, 1, 100)
y2 = np.random.normal(5, 1, 100)

# Grafica el primer conjunto de datos
plt.scatter(x1, y1, c='red')

# Grafica el segundo conjunto de datos
plt.scatter(x2, y2, c='blue')

# Calcula la media de los valores de x e y de ambos conjuntos
x_medio = (np.mean(x1) + np.mean(x2)) / 2
y_medio = (np.mean(y1) + np.mean(y2)) / 2

# Grafica el punto en el medio
plt.scatter(x_medio, y_medio, c='green', s=100)

# Define una función para calcular la distancia euclidiana utilizando el teorema de Pitágoras
def distancia_pitagoras(x1, y1, x2, y2):
    distancia = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return distancia

# Calcula la distancia entre el punto medio y cada punto de la primera dispersión
distancias1 = []
for i in range(len(x1)):
    distancia = distancia_pitagoras(x1[i], y1[i], x_medio, y_medio)
    distancias1.append(distancia)

# Calcula la distancia entre el punto medio y cada punto de la segunda dispersión
distancias2 = []
for i in range(len(x2)):
    distancia = distancia_pitagoras(x2[i], y2[i], x_medio, y_medio)
    distancias2.append(distancia)

# Determina a qué dispersión pertenece el punto medio
if min(distancias1) < min(distancias2):
    pertenece_a = "disposición 1"
else:
    pertenece_a = "disposición 2"

# Muestra a qué dispersión pertenece el punto medio
print("El punto medio pertenece a la", pertenece_a)

# Muestra la gráfica
plt.show()
