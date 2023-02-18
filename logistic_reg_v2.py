import numpy as np
import matplotlib.pyplot as plt

# Generar datos de ejemplo
np.random.seed(0)
n = 100
x = np.linspace(0, 5, n)
y = 1 / (1 + np.exp(-(2*x-5)))
y = np.random.binomial(1, y)

# Ajustar la regresión logística con gradiente descendente
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def cost_function(theta, X, y):
    h = sigmoid(X.dot(theta))
    J = -1/m * (y.T.dot(np.log(h)) + (1-y).T.dot(np.log(1-h)))
    grad = 1/m * X.T.dot(h - y)
    return J, grad

def gradient_descent(X, y, alpha, iterations, theta_init):
    J_history = []
    theta = theta_init
    for i in range(iterations):
        J, grad = cost_function(theta, X, y)
        theta = theta - alpha * grad
        J_history.append(J)
    return theta, J_history

m = len(y)
X = np.vstack((np.ones(n), x)).T
y = y.reshape((m, 1))
theta_init = np.zeros((2, 1))

# Gradiente muy bueno
theta1, J1 = gradient_descent(X, y, 10, 100, theta_init)

# Gradiente bueno
theta2, J2 = gradient_descent(X, y, 1, 100, theta_init)

# Gradiente mala
theta3, J3 = gradient_descent(X, y, 0.1, 100, theta_init)

# Gradiente muy mala
theta4, J4 = gradient_descent(X, y, 0.01, 100, theta_init)

# Graficar los resultados
plt.figure(figsize=(8,6))
plt.scatter(x, y, c='b')
plt.plot(x, sigmoid(X.dot(theta1)), 'r', label='Gradiente muy bueno')
plt.plot(x, sigmoid(X.dot(theta2)), 'g', label='Gradiente bueno')
plt.plot(x, sigmoid(X.dot(theta3)), 'y', label='Gradiente mala')
plt.plot(x, sigmoid(X.dot(theta4)), 'k', label='Gradiente muy mala')
plt.legend(loc='best')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
