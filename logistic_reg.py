# Python de regresión logística con gradiente descendente para diferentes situaciones de gradiente.
import numpy as np
import matplotlib.pyplot as plt

# Generar datos sintéticos
np.random.seed(42)

# Datos de clase 0
X0 = np.random.multivariate_normal(mean=[2, 2], cov=[[1, 0], [0, 1]], size=50)
y0 = np.zeros(50)

# Datos de clase 1
X1 = np.random.multivariate_normal(mean=[-2, -2], cov=[[1, 0], [0, 1]], size=50)
y1 = np.ones(50)

# Concatenar datos y etiquetas
X = np.concatenate([X0, X1])
y = np.concatenate([y0, y1])

# Agregar columna de unos a X para el término de sesgo
X = np.hstack([X, np.ones((100, 1))])

# Visualizar datos
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.show()
# ----------------------------------------------------------------
# Función sigmoide
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Función de costo
def cost_function(X, y, theta):
    h = sigmoid(X.dot(theta))
    J = (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()
    return J

# Gradiente descendente
def gradient_descent(X, y, theta, alpha, num_iters):
    m = len(y)
    J_history = []

    for i in range(num_iters):
        h = sigmoid(X.dot(theta))
        theta = theta - alpha * (X.T.dot(h - y)) / m
        J_history.append(cost_function(X, y, theta))

    return theta, J_history

# Inicializar parámetros
theta = np.zeros(3)

# Entrenamiento con gradiente descendente
theta, J_history = gradient_descent(X, y, theta, alpha=0.1, num_iters=1000)

# Visualizar resultado
plt.plot(J_history)
plt.xlabel('Iteraciones')
plt.ylabel('Costo')
plt.show()
