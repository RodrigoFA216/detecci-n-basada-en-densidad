import math

class Neurona:
    def __init__(self, w1, w2, theta):
        self.w1 = w1
        self.w2 = w2
        self.theta = theta

    def activacion(self, x1, x2):
        z = (self.w1 * x1) + (self.w2 * x2) - self.theta
        y = 1 / (1 + math.exp(-z))
        return y

n = Neurona(0.5, 0.5, 0.2)
salida = n.activacion(1, 0.5)
print(salida)
