from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import json
import os

app = Flask(__name__)
CORS(app)

# ✅ Load model
model_path = os.path.join(os.path.dirname(__file__), "model", "crop_disease_model.h5")
model = tf.keras.models.load_model(model_path)

# ✅ Load class names (now inside model folder)
class_names_path = os.path.join(os.path.dirname(__file__), "model", "class_names.txt")
with open(class_names_path, "r") as f:
    class_names = [line.strip() for line in f.readlines()]

# ✅ Load advisory data
advisory_path = os.path.join(os.path.dirname(__file__), "advisory.json")
with open(advisory_path, "r") as f:
    advisory_data = json.load(f)

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    try:
        # ✅ Preprocess image
        image = Image.open(file.stream).convert("RGB")
        image = image.resize((224, 224))
        image_array = np.array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)

        # ✅ Predict
        predictions = model.predict(image_array)
        predicted_index = np.argmax(predictions[0])
        predicted_class = class_names[predicted_index]

        # ✅ Get advisory for predicted disease
        advisory = advisory_data.get(predicted_class, {
            "symptoms": "Not available",
            "cause": "Not available",
            "treatment": "Not available",
            "prevention": "Not available"
        })

        return jsonify({
            "predicted_class": predicted_class,
            "advisory": advisory
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/")
def home():
    return "🌱 Smart Crop Disease Detection API is running!"


if __name__ == "__main__":
    app.run(debug=True)
