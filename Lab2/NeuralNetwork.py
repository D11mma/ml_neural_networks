import os
import logging

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

tf.get_logger().setLevel('ERROR')
logging.getLogger('tensorflow').setLevel(logging.ERROR)
tf.config.optimizer.set_experimental_options({"remapper": False})

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, SimpleRNN, Input, Concatenate

def generate_sphere_data(n_samples=3000):
    x_raw = np.random.uniform(-4, 4, n_samples)
    y_raw = np.random.uniform(-4, 4, n_samples)
    mask = x_raw ** 2 + y_raw ** 2 <= 15.9
    x, y = x_raw[mask], y_raw[mask]
    z = np.sqrt(16 - x ** 2 - y ** 2)
    return np.column_stack((x, y)), z

X, Y = generate_sphere_data(5000)
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
Y_train, Y_test = Y[:split], Y[split:]

grid_range = np.linspace(-4, 4, 50)
gx, gy = np.meshgrid(grid_range, grid_range)
gz_ideal = 16 - gx ** 2 - gy ** 2
gz_ideal[gz_ideal < 0] = np.nan
gz_ideal = np.sqrt(gz_ideal)

print("Відображення основної функції...")
fig_ideal = plt.figure(figsize=(8, 6))
ax_ideal = fig_ideal.add_subplot(111, projection='3d')
ax_ideal.plot_surface(gx, gy, gz_ideal, cmap='Reds', alpha=0.8)
ax_ideal.set_title("Оригінальна функція: z = sqrt(16 - x^2 - y^2)")
ax_ideal.set_xlabel("Вісь X")
ax_ideal.set_ylabel("Вісь Y")
ax_ideal.set_zlabel("Вісь Z")
plt.show()

def create_cascade_model(layers_config):
    inputs = Input(shape=(2,))
    curr = inputs
    for n in layers_config:
        h = Dense(n, activation='relu')(curr)
        curr = Concatenate()([curr, h])
    return Model(inputs=inputs, outputs=Dense(1)(curr))

models_dict = {
    "FF_10_нейронів": Sequential([Input(shape=(2,)), Dense(10, activation='relu'), Dense(1)]),
    "FF_20_нейронів": Sequential([Input(shape=(2,)), Dense(20, activation='relu'), Dense(1)]),
    "Cascade_20_нейронів": create_cascade_model([20]),
    "Cascade_2x10_нейронів": create_cascade_model([10, 10]),
    "Elman_15_нейронів": Sequential([Input(shape=(2, 1)), SimpleRNN(15, activation='relu'), Dense(1)]),
    "Elman_3x5_нейронів": Sequential([
        Input(shape=(2, 1)), SimpleRNN(5, activation='relu', return_sequences=True),
        SimpleRNN(5, activation='relu', return_sequences=True), SimpleRNN(5, activation='relu'), Dense(1)
    ])
}

results = []

for name, model in models_dict.items():
    print(f"Тренування: {name}...")

    if "Elman" in name:
        X_tr, X_te = X_train.reshape(-1, 2, 1), X_test.reshape(-1, 2, 1)
        X_grid_flat = np.column_stack((gx.ravel(), gy.ravel())).reshape(-1, 2, 1)
    else:
        X_tr, X_te = X_train, X_test
        X_grid_flat = np.column_stack((gx.ravel(), gy.ravel()))

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    history = model.fit(X_tr, Y_train, epochs=120, batch_size=32, validation_data=(X_te, Y_test), verbose=0)

    Y_pred = model.predict(X_te, verbose=0).flatten()
    rel_error = np.mean(np.abs((Y_test - Y_pred) / (Y_test + 1e-7))) * 100
    r2 = r2_score(Y_test, Y_pred)
    results.append({"Model": name, "Error": rel_error, "R2": r2})

    fig = plt.figure(figsize=(16, 6))
    fig.suptitle(f"Модель: {name} | Помилка: {rel_error:.2f}%", fontsize=14)

    Z_pred = model.predict(X_grid_flat, verbose=0).reshape(gx.shape)
    Z_pred[np.isnan(gz_ideal)] = np.nan
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax1.plot_wireframe(gx, gy, gz_ideal, color='black', alpha=0.1)
    ax1.plot_surface(gx, gy, Z_pred, cmap='Greens', alpha=0.7)
    ax1.set_title("Результат апроксимації (3D)")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(history.history['loss'], label='Навчання (MSE)')
    ax2.plot(history.history['val_loss'], label='Валідація (MSE)')
    ax2.set_title('Графік навчання (Помилка MSE)')
    ax2.set_xlabel('Епохи')
    ax2.set_ylabel('Значення помилки')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

print("\n" + "-" * 65)
print(f"{'Назва моделі':<25} | {'Помилка (%)':<15} | {'Коеф. R2'}")
print("-" * 65)
for res in results:
    print(f"{res['Model']:<25} | {res['Error']:<15.2f} | {res['R2']:.4f}")