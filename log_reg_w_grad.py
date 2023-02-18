import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Cargar el conjunto de datos Iris
iris = load_iris()

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3, random_state=42)

# Escalar los datos
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Definir la función sigmoide
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Definir la función de costo y su gradiente
def cost_function(theta, X, y):
    m = len(y)
    h = sigmoid(X.dot(theta))
    J = (1 / m) * (-y.T.dot(np.log(h)) - (1 - y).T.dot(np.log(1 - h)))
    grad = (1 / m) * X.T.dot(h - y)
    return J, grad

# Definir la función de entrenamiento
def train(X, y, theta, alpha, iterations):
    J_history = []
    for i in range(iterations):
        J, grad = cost_function(theta, X, y)
        theta = theta - alpha * grad
        J_history.append(J)
    return theta, J_history

# Inicializar los parámetros
theta = np.zeros(X_train.shape[1])
alpha = 0.1
iterations = 1000

# Entrenar el modelo
theta, J_history = train(X_train, y_train, theta, alpha, iterations)

# Realizar predicciones en el conjunto de prueba
y_pred = sigmoid(X_test.dot(theta))
y_pred = np.where(y_pred >= 0.5, 1, 0)

# Evaluar el modelo
accuracy = np.mean(y_pred == y_test)
print("Exactitud: ", accuracy)

# Graficar la función de costo
plt.plot(J_history)
plt.xlabel("Iteraciones")
plt.ylabel("Costo")
plt.show()
