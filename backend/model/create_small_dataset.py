import os
import random
import shutil

# Original dataset path (nested with train_set/test_set inside each crop)
original_dataset = r"C:\Users\Prath\Downloads\smart-crop-app\backend\model\dataset\Dataset for Crop Pest and Disease Detection\CCMT Dataset-Augmented"

# Small dataset path
small_dataset = r"C:\Users\Prath\Downloads\smart-crop-app\backend\model\dataset\CCMT_small"
os.makedirs(small_dataset, exist_ok=True)

# Number of images per class
num_images_per_class = 50

# Train/Test ratio
train_ratio = 0.8

# Create train/test folders
train_base = os.path.join(small_dataset, "train_set")
test_base = os.path.join(small_dataset, "test_set")
os.makedirs(train_base, exist_ok=True)
os.makedirs(test_base, exist_ok=True)

# Iterate crops
for crop in os.listdir(original_dataset):
    crop_path = os.path.join(original_dataset, crop)
    if not os.path.isdir(crop_path):
        continue

    # Iterate split folders inside crop (train_set/test_set)
    for split in os.listdir(crop_path):
        split_path = os.path.join(crop_path, split)
        if not os.path.isdir(split_path):
            continue

        # Iterate disease classes
        for disease_class in os.listdir(split_path):
            src_folder = os.path.join(split_path, disease_class)
            if not os.path.isdir(src_folder):
                continue

            # List all image files
            images = [img for img in os.listdir(src_folder)
                      if os.path.isfile(os.path.join(src_folder, img))]
            if not images:
                continue

            # Randomly select images
            selected_images = random.sample(images, min(len(images), num_images_per_class))

            # Split train/test
            split_index = int(len(selected_images) * train_ratio)
            train_images = selected_images[:split_index]
            test_images = selected_images[split_index:]

            # Create destination folders
            train_folder = os.path.join(train_base, crop, disease_class)
            test_folder = os.path.join(test_base, crop, disease_class)
            os.makedirs(train_folder, exist_ok=True)
            os.makedirs(test_folder, exist_ok=True)

            # Copy images
            for img in train_images:
                shutil.copy(os.path.join(src_folder, img), os.path.join(train_folder, img))
            for img in test_images:
                shutil.copy(os.path.join(src_folder, img), os.path.join(test_folder, img))

print("✅ Small dataset with 50 images per class created successfully!")
