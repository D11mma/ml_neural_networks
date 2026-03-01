import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

X = np.array([
    [0, 0, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 1, 0],
    [0, 0, 1, 1],
    [0, 1, 0, 0],
    [0, 1, 0, 1],
    [0, 1, 1, 0],
    [0, 1, 1, 1],
    [1, 0, 0, 0],
    [1, 0, 0, 1],
    [1, 0, 1, 0],
    [1, 0, 1, 1],
    [1, 1, 0, 0],
    [1, 1, 0, 1],
    [1, 1, 1, 0],
    [1, 1, 1, 1]
], dtype=np.float32)

y = np.array([
    [0], [1], [1], [0],
    [1], [0], [0], [1],
    [1], [0], [0], [1],
    [0], [1], [1], [0]
], dtype=np.float32)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(4,)),
    tf.keras.layers.Dense(16, activation="relu"),
    tf.keras.layers.Dense(8, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss='binary_crossentropy',metrics=['accuracy']
)

print("Починаємо навчання...")
history = model.fit(X, y, epochs=500, verbose=0)

loss, accuracy = model.evaluate(X, y, verbose=0)
print(f"\nКінцева точність: {accuracy * 100:.2f}%")

print("\nПеревірка передбачень:")
predictions = model.predict(X)
for i in range(len(X)):
    pred_label = 1 if predictions[i] > 0.5 else 0
    print(f"Вхід: {X[i]} -> Результат: {pred_label} (ймовірність: {predictions[i][0]:.4f})")

plt.figure(figsize=(8, 5))
plt.plot(history.history['accuracy'], label='Accuracy')
plt.title('Точність під час навчання')
plt.xlabel('Епоха')
plt.ylabel('Точність')
plt.legend()
plt.savefig('accuracy_chart.png', dpi=100, bbox_inches='tight')
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], color='red', label='Loss')
plt.title('Втрати під час навчання')
plt.xlabel('Епоха')
plt.ylabel('Помилка (Loss)')
plt.legend()
plt.savefig('loss_chart.png', dpi=100, bbox_inches='tight')
plt.show()
