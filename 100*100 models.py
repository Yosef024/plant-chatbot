import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D,
    Permute, Reshape,
    LSTM,
    Dense, Dropout
)
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.callbacks import ReduceLROnPlateau
from sklearn.metrics import confusion_matrix,classification_report
import numpy as np
from PIL import Image
import os
import seaborn as sns
from collections import Counter
import shutil
import random
import tensorflow as tf
print("TF version:", tf.__version__)
print("GPU devices:", tf.config.list_physical_devices('GPU'))
print("Built with CUDA:", tf.test.is_built_with_cuda())

dataset = 'type'



# Now proceed with ImageDataGenerator as usual
imgsize = (100, 100)
batch = 64
epochs = 120

datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    brightness_range=[0.6, 1.3],
    horizontal_flip=True,
    validation_split=0.1,
)

train = datagen.flow_from_directory(
    dataset, target_size=imgsize, batch_size=batch, class_mode="sparse", subset="training"
)

val = datagen.flow_from_directory(
    dataset, target_size=imgsize, batch_size=batch, class_mode="sparse", subset="validation"
)

model = Sequential([
    # ─── Convolutional Backbone ───────────────────────────
    Conv2D(32, (3, 3), activation='swish', input_shape=(100,100,3)),
    MaxPooling2D((2, 2)),

    Conv2D(64, (3, 3), activation='swish'),
    MaxPooling2D((2, 2)),

    Conv2D(128, (3, 3), activation='swish'),
    MaxPooling2D((2, 2)),
    # At this point: tensor shape = (batch, H_p, W_p, 128)

    # ─── Prepare for LSTM ─────────────────────────────────
    # 1) Permute so that channels (128) become the "time" axis
    Permute((3, 1, 2)),
    # new shape = (batch, 128, H_p, W_p)

    # 2) Flatten spatial dims into features per time‐step
    Reshape((128, -1)),
    # final shape = (batch, time_steps=128, features=H_p*W_p)

    # ─── Sequence Model ──────────────────────────────────
    LSTM(512, return_sequences=False, name='inter_filter_lstm'),
    # output shape = (batch, 64)

    # ─── Classifier Head ─────────────────────────────────
    Dense(512, activation='swish'),
    Dropout(0.4),

    Dense(256, activation='swish'),
    Dropout(0.4),

    Dense(128, activation='swish'),
    Dropout(0.4),

    Dense(64, activation='swish'),
    Dropout(0.4),

    Dense(len(train.class_indices), activation='softmax')
])
print(train.class_indices)
model.compile(
    optimizer='adamax',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
early_stopping = EarlyStopping(monitor='loss', patience=15, restore_best_weights=True)
history = model.fit(
    train,
    epochs=epochs,
    validation_data=val,
    callbacks=[early_stopping,reduce_lr]
)
test_loss, test_acc = model.evaluate(val)
print(f'Test Accuracy: {test_acc:.4f}')

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the TensorFlow Lite model
tflite_model_path = 'plant type edited lstm.tflite'
with open(tflite_model_path, 'wb') as f:
    f.write(tflite_model)

print(f'Model has been converted to TensorFlow Lite format and saved as {tflite_model_path}.')

model.save('plant type model lstm.h5')

plt.figure(figsize=(12, 4))

# Plot Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')

# Plot Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')

plt.show()
y_true = val.classes  # Actual labels
y_pred_probs = model.predict(val)  # Model predictions (probabilities)
y_pred = np.argmax(y_pred_probs, axis=1)  # Convert probabilities to class labels

# Compute Confusion Matrix
conf_matrix = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=train.class_indices.keys(), yticklabels=train.class_indices.keys())
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=train.keys()))




