import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os

# Load the model
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "crop_disease_model.h5")
model = tf.keras.models.load_model(MODEL_PATH)

# Load class names in correct order
CLASS_NAMES_PATH = os.path.join(os.path.dirname(BASE_DIR), "utils", "class_names.txt")
with open(CLASS_NAMES_PATH, "r") as f:
    class_names = [line.strip() for line in f.readlines() if line.strip()]

# --- Ensure same preprocessing used during training ---
IMG_SIZE = (224, 224)

def preprocess_image(image_bytes):
    """Preprocess the uploaded image properly before prediction"""
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize(IMG_SIZE)
    image_array = np.array(image).astype("float32") / 255.0  # Normalize
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    return image_array


def predict_disease(image_bytes):
    """Predict the disease class for the given image"""
    processed_image = preprocess_image(image_bytes)

    # Model prediction
    predictions = model.predict(processed_image)
    predicted_index = np.argmax(predictions[0])
    confidence = float(np.max(predictions[0]))

    predicted_class = class_names[predicted_index]

    return predicted_class, confidence
