# ablation_study.py
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import os

dataset_path = r"C:\Users\Prath\Downloads\smart-crop-app\backend\model\dataset\CCMT_small"

# Image sizes to test
image_sizes = [(128,128), (224,224), (256,256)]
batch_size = 32
epochs = 10

results = []

for size in image_sizes:
    print(f"\n--- Training with image size: {size} ---")
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
    
    train_data = datagen.flow_from_directory(
        dataset_path,
        target_size=size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )
    
    val_data = datagen.flow_from_directory(
        dataset_path,
        target_size=size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )
    
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(size[0], size[1], 3)),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(train_data.num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    history = model.fit(train_data, validation_data=val_data, epochs=epochs, verbose=1)
    
    val_acc = history.history['val_accuracy'][-1]
    results.append((size, val_acc))
    print(f"Validation Accuracy for size {size}: {val_acc:.4f}")

# Plot results
sizes = [f"{s[0]}x{s[1]}" for s,_ in results]
accuracies = [acc for _,acc in results]

plt.bar(sizes, accuracies, color='#4CAF50')
plt.title("Ablation Study: Effect of Image Size on Validation Accuracy")
plt.ylabel("Validation Accuracy")
plt.xlabel("Image Size")
plt.ylim(0,1)
plt.show()
