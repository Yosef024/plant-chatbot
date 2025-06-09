import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report

# Load the H5 model
model = tf.keras.models.load_model('plant type lstm full 2 (2).h5')

# Set dataset path and parameters
dataset_path = 'resized splitted type/train'
img_size = (224, 224)

# Initialize ImageDataGenerator for validation data
datagen = ImageDataGenerator(rescale=1.0 / 255)

# Create validation data generator with batch size of 1
val_generator = datagen.flow_from_directory(

    'splitted type/test',
    target_size=img_size,
    batch_size=1,  # Set batch size to 1
    class_mode='sparse',
    shuffle=False  # Important: keep data in order for evaluation
)

# Initialize lists to store true labels and predictions
true_labels = []
predicted_labels = []

# Iterate over validation data one image at a time
for i in range(len(val_generator)):
    # Get the next image and label
    batch_images, batch_labels = val_generator[i]

    # Make prediction using the loaded model
    predictions = model.predict(batch_images, verbose=0)

    # Convert model output to predicted label
    predicted_label = np.argmax(predictions, axis=1)[0]

    # Store true label and prediction
    true_labels.append(batch_labels[0])
    predicted_labels.append(predicted_label)

# Convert lists to numpy arrays
true_labels = np.array(true_labels)
predicted_labels = np.array(predicted_labels)

# Compute the confusion matrix
conf_matrix = confusion_matrix(true_labels, predicted_labels)

# Retrieve class labels
class_labels = list(val_generator.class_indices.keys())

# Plot the confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap="Blues",
            xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

# Print the classification report
print(classification_report(true_labels, predicted_labels, target_names=class_labels, digits=4))
