import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# -------------------------------
# Paths
# -------------------------------
dataset_path = r"C:\Users\Prath\Downloads\smart-crop-app\backend\model\dataset\CCMT_small"
model_path = os.path.join(os.path.dirname(__file__), "crop_disease_small_model.h5")
label_path = os.path.join(os.path.dirname(__file__), "label_classes_small.npy")
checkpoint_path = os.path.join(os.path.dirname(__file__), "checkpoints_small")

os.makedirs(checkpoint_path, exist_ok=True)

# -------------------------------
# Image & Training parameters
# -------------------------------
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 8      # small batch size since dataset is tiny
EPOCHS = 15         # small number of epochs for fast training

# -------------------------------
# Data preparation
# -------------------------------
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_data = datagen.flow_from_directory(
    os.path.join(dataset_path, "train_set"),
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True,
    subset='training'
)

val_data = datagen.flow_from_directory(
    os.path.join(dataset_path, "train_set"),
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False,
    subset='validation'
)

# Save class labels
np.save(label_path, train_data.class_indices)
print(f"✅ Class labels saved to {label_path}")

# -------------------------------
# Model architecture
# -------------------------------
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(train_data.num_classes, activation='softmax')
])

# -------------------------------
# Compile model
# -------------------------------
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# -------------------------------
# Callbacks
# -------------------------------
checkpoint = ModelCheckpoint(
    filepath=os.path.join(checkpoint_path, "epoch_{epoch:02d}_valacc_{val_accuracy:.2f}.h5"),
    save_best_only=True,
    verbose=1
)

earlystop = EarlyStopping(
    monitor='val_accuracy',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

# -------------------------------
# Train model
# -------------------------------
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    callbacks=[checkpoint, earlystop]
)

# -------------------------------
# Save final model
# -------------------------------
model.save(model_path)
print(f"✅ Final model saved to: {model_path}")
print(f"📂 Checkpoints saved in: {checkpoint_path}")
