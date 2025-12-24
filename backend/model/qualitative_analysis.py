import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Paths
model_path = os.path.join(os.path.dirname(__file__), "crop_disease_small_model.h5")
label_path = os.path.join(os.path.dirname(__file__), "label_classes_small.npy")
dataset_path = os.path.join(os.path.dirname(__file__), "dataset", "CCMT_small")

# Load model
model = tf.keras.models.load_model(model_path)
print(f"✅ Model loaded from {model_path}")

# Load class labels
class_indices = np.load(label_path, allow_pickle=True).item()
class_names = [k for k, v in sorted(class_indices.items(), key=lambda item: item[1])]
print(f"✅ Class labels loaded: {class_names}")

# Image parameters
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32

# Test data generator
test_datagen = ImageDataGenerator(rescale=1./255)

test_data = test_datagen.flow_from_directory(
    os.path.join(dataset_path, "test_set"),
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

# Function to display predictions
def display_predictions(num_images=10):
    plt.figure(figsize=(18, 6))
    misclassified_count = 0

    # Use iterator for DirectoryIterator
    test_iterator = iter(test_data)

    for i in range(num_images):
        img, label = next(test_iterator)
        pred = model.predict(img)
        pred_class = np.argmax(pred, axis=1)[0]
        true_class = np.argmax(label, axis=1)[0]

        plt.subplot(2, (num_images+1)//2, i+1)
        plt.imshow(img[0])
        plt.axis('off')
        color = 'green' if pred_class == true_class else 'red'
        plt.title(f"T: {class_names[true_class]}\nP: {class_names[pred_class]}", color=color)

        if pred_class != true_class:
            misclassified_count += 1

    plt.tight_layout()
    plt.show()
    print(f"Misclassified images in sample: {misclassified_count}/{num_images}")

# Run qualitative analysis
display_predictions(num_images=12)
