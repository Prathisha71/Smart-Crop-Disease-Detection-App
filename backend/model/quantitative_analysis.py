import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# Paths
dataset_path = r"C:\Users\Prath\Downloads\smart-crop-app\backend\model\dataset\CCMT_small\test_set"
model_path = r"C:\Users\Prath\Downloads\smart-crop-app\backend\model\crop_disease_small_model.h5"
label_path = r"C:\Users\Prath\Downloads\smart-crop-app\backend\model\label_classes_small.npy"

# Load model & class labels
model = tf.keras.models.load_model(model_path)
class_indices = np.load(label_path, allow_pickle=True).item()
class_names = list(class_indices.keys())

# Image parameters
IMG_HEIGHT, IMG_WIDTH = 224, 224
BATCH_SIZE = 32

# Load test data
datagen = ImageDataGenerator(rescale=1./255)
test_data = datagen.flow_from_directory(
    dataset_path,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# Predict
predictions = model.predict(test_data)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = test_data.classes

# 1️⃣ Classification Report
report = classification_report(true_classes, predicted_classes, target_names=class_names)
print("\n📊 Classification Report:\n")
print(report)

# 2️⃣ Accuracy Score
accuracy = accuracy_score(true_classes, predicted_classes)
print(f"\n✅ Test Set Accuracy: {accuracy*100:.2f}%")

# 3️⃣ Confusion Matrix
cm = confusion_matrix(true_classes, predicted_classes)

# 4️⃣ Heatmap Visualization
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted Class')
plt.ylabel('Actual Class')
plt.title('Confusion Matrix Heatmap')
plt.xticks(rotation=45)
plt.yticks(rotation=45)
plt.tight_layout()
plt.savefig('confusion_matrix_heatmap.png', dpi=300)
plt.show()

# 5️⃣ Optional: Top misclassified classes
misclassified = np.where(predicted_classes != true_classes)[0]
print(f"\n⚠️ Total Misclassified Images: {len(misclassified)}")
if len(misclassified) > 0:
    print("Some examples of misclassified classes:")
    for i in misclassified[:10]:
        print(f"Image: {test_data.filenames[i]} | True: {class_names[true_classes[i]]} | Predicted: {class_names[predicted_classes[i]]}")
