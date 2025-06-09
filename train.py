import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout,Permute,Reshape,LSTM,RNN, LSTMCell
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.callbacks  import ReduceLROnPlateau
from sklearn.metrics import confusion_matrix,classification_report
import numpy as np
from PIL import Image
import os
import seaborn as sns
from collections import Counter
import shutil
import random

dataset = '/kaggle/input/type-full/splitted type/train'

imgsize = (224, 224)
batch = 64
epochs = 50

datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=30,
    #width_shift_range=0.2,
    #height_shift_range=0.2,
    zoom_range=0.2,
    brightness_range=[0.8, 1.2],
    #horizontal_flip=True,
    #validation_split=0.1,
)
datagentest = ImageDataGenerator(
    rescale=1.0 / 255,
    #rotation_range=30,
    #width_shift_range=0.2,
    #height_shift_range=0.2,
    #zoom_range=0.2,
    #brightness_range=[0.8, 1.2],
    #horizontal_flip=True,
    #validation_split=0.1,
)

train = datagen.flow_from_directory(
    '/kaggle/input/type-full/splitted type/train', target_size=imgsize, batch_size=batch, class_mode="sparse",
)

val = datagentest.flow_from_directory(
    '/kaggle/input/type-full/splitted type/test', target_size=imgsize, batch_size=batch, class_mode="sparse",
)

model = Sequential([
    # ── Block 1 ──
    Conv2D(32, (3, 3),
           activation='relu',
           padding='same',
           input_shape=(224, 224, 3)),
    BatchNormalization(),
    MaxPooling2D((2, 2), padding='same'),  # 224×224×32 -> 112×112×32

    # ── Block 2 ──
    Conv2D(64, (3, 3),
           activation='relu',
           padding='same'),
    BatchNormalization(),
    MaxPooling2D((2, 2), padding='same'),  # 112×112×64 -> 56×56×64

    # ── Block 3 ──
    Conv2D(128, (3, 3),
           activation='relu',
           padding='same'),
    BatchNormalization(),
    MaxPooling2D((2, 2), padding='same'),  # 56×56×128 -> 14×14×128
    
    #Flatten(),
    # ── Reshape for LSTM ──
    # Current tensor shape: (batch, 14, 14, 128)
    # Reshape to: (batch, 14*14, 128) = (batch, 361, 128)
    Reshape((-1, 128)),

    # ── High-level LSTM (TFLite supports this) ──
    LSTM(
        units=3000,
        return_sequences=False,
        unroll=True,
        name='inter_filter_lstm'
    ),

    # ── Dense “head” ──
    Dense(512, activation='relu'),
    Dropout(0.4),
    Dense(256, activation='relu'),
    Dropout(0.4),
    Dense(128, activation='relu'),
    Dropout(0.4),
    Dense(64, activation='relu'),
    Dropout(0.4),

    # ── Final softmax ──
    # Replace `NUM_CLASSES` with the actual number of classes in your dataset
    Dense(len(train.class_indices), activation='softmax')
])
print(train.class_indices)

model.compile(
    optimizer='adamax',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=5e-4)
early_stopping = EarlyStopping(monitor='loss', patience=8, restore_best_weights=True)
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
tflite_model_path = 'plant disease lstm full 2.tflite'
with open(tflite_model_path, 'wb') as f:
    f.write(tflite_model)

print(f'Model has been converted to TensorFlow Lite format and saved as {tflite_model_path}.')

model.save('plant disease lstm full 2.h5')

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
print(classification_report(y_true, y_pred, target_names=train.class_indices.keys()))
