import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.layers import Input, Dense
from keras.models import Model

# Cargar los datos en un DataFrame de Pandas
data = pd.read_csv("data.csv")

# Dividir los datos en entrenamiento y prueba
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Definir la arquitectura de la red neuronal autoencoder
input_layer = Input(shape=(train_data.shape[1],))
encoded = Dense(20, activation='relu')(input_layer)
decoded = Dense(train_data.shape[1], activation='sigmoid')(encoded)
autoencoder = Model(input_layer, decoded)

# Compilar y entrenar el modelo
autoencoder.compile(optimizer='adam', loss='mean_squared_error')
autoencoder.fit(train_data, train_data, epochs=50, batch_size=32, shuffle=True)

# Evaluar el modelo en los datos de prueba
reconstructed_data = autoencoder.predict(test_data)
mse = np.mean(np.power(test_data - reconstructed_data, 2), axis=1)
threshold = np.mean(mse) + 3 * np.std(mse)
outliers = np.where(mse > threshold)[0]

# Visualizar los resultados
plt.scatter(range(test_data.shape[0]), mse, color='blue', label='mse')
plt.axhline(y=threshold, color='red', label='threshold')
plt.scatter(outliers, mse[outliers], color='red', label='outliers')
plt.legend()
plt.show()


# # Generate random data
# np.random.seed(0)
# data = np.random.randn(100, 5)

# # Convert data to DataFrame
# df = pd.DataFrame(data, columns=['A', 'B', 'C', 'D', 'E'])

# Download the Titanic dataset
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
data = pd.read_csv(url)

# Print the first 5 rows of the data
print(data.head())