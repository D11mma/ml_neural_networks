import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
from PIL import Image, ImageOps

print("Завантаження даних...")
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_train_norm = x_train.astype("float32") / 255
x_test_norm = x_test.astype("float32") / 255
x_train_flat = x_train_norm.reshape(-1, 784)
x_test_flat = x_test_norm.reshape(-1, 784)

y_train_cat = keras.utils.to_categorical(y_train, 10)
y_test_cat = keras.utils.to_categorical(y_test, 10)

print("Створення архітектури...")
model = keras.Sequential([
    layers.Input(shape=(784,)),
    layers.Dense(256, activation="relu"),
    layers.Dropout(0.2),
    layers.Dense(128, activation="relu"),
    layers.Dense(10, activation="softmax")
])

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

print("Навчання моделі...")
history = model.fit(x_train_flat, y_train_cat, epochs=10, batch_size=64, validation_split=0.1, verbose=2)

print("\nТестування на тестових даних MNIST...")
predictions = model.predict(x_test_flat, verbose=0)
y_pred_classes = np.argmax(predictions, axis=1)


plt.figure(figsize=(10, 3))
for i in range(5):
    plt.subplot(1, 5, i + 1)
    plt.imshow(x_test[i], cmap='gray')
    plt.title(f"Реально: {y_test[i]}\nПрогноз: {y_pred_classes[i]}")
    plt.axis('off')
plt.tight_layout()
plt.show()

print("\nОцінка правильності роботи...")

acc = accuracy_score(y_test, y_pred_classes)
prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred_classes, average='weighted')

print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F-Score:   {f1:.4f}")

plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred_classes)
sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrBr')
plt.title('Матриця помилок (Confusion Matrix)')
plt.ylabel('Справжні класи')
plt.xlabel('Передбачені класи')
plt.show()

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Точність (Accuracy)')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Помилка (Loss)')
plt.legend()
plt.show()

def test_my_digit(img_path):
    print(f"\nТестування власної цифри: {img_path}")
    try:
        img = Image.open(img_path).convert('L')
        img = img.resize((28, 28))
        img = ImageOps.invert(img)

        img_arr = np.array(img).astype("float32") / 255
        img_arr = img_arr.reshape(1, 784)

        res = model.predict(img_arr, verbose=0)
        digit = np.argmax(res)
        conf = np.max(res)

        plt.imshow(img, cmap='gray')
        plt.title(f"Мій малюнок: {digit} (Впевненість: {conf * 100:.2f}%)")
        plt.show()
    except Exception as e:
        print(f"Файл не знайдено або помилка: {e}")

test_my_digit('Number.png')
