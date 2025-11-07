import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

# ===========================
# Dataset de entrenamiento para puerta XOR
# ===========================
x_train = np.array([[0,0], [0,1], [1,0], [1,1]], dtype="float32")
y_train = np.array([[0], [1], [1], [0]], dtype="float32")

# ===========================
# Modelo MLP (Multilayer Perceptron)
# ===========================
model = keras.Sequential()
model.add(layers.Dense(2, input_dim=2, activation='relu'))   # Capa oculta
model.add(layers.Dense(1, activation='sigmoid'))             # Capa de salida

# ===========================
# Configuración del modelo
# ===========================
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.1),
    loss='mean_squared_error',
    metrics=['accuracy']
)

# ===========================
# Entrenamiento
# ===========================
fit_history = model.fit(x_train, y_train, epochs=50, batch_size=4, verbose=1)

# ===========================
# Gráfico de pérdida
# ===========================
loss_curve = fit_history.history['loss']
accuracy_curve = fit_history.history['accuracy']

plt.figure(figsize=(8, 4))
plt.plot(loss_curve, label='Pérdida (Loss)')
plt.plot(accuracy_curve, label='Precisión (Accuracy)')
plt.xlabel('Épocas')
plt.ylabel('Valor')
plt.title('Resultado del Entrenamiento (XOR)')
plt.legend()
plt.show()

# ===========================
# Recuperar pesos y sesgos
# ===========================
weights_HL, biases_HL = model.layers[0].get_weights()  # Capa oculta
weights_OL, biases_OL = model.layers[1].get_weights()  # Capa de salida

print("\nPesos Capa Oculta:\n", weights_HL)
print("Sesgos Capa Oculta:\n", biases_HL)
print("\nPesos Capa de Salida:\n", weights_OL)
print("Sesgos Capa de Salida:\n", biases_OL)

# ===========================
# Predicciones del modelo
# ===========================
prediccion = model.predict(x_train)
print("\nPredicciones:\n", prediccion)
print("\nEntradas:\n", x_train)
print("\nSalidas Esperadas:\n", y_train)

