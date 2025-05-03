import os
import shutil
import random
from glob import glob
from tqdm import tqdm

# Define paths
DATA_DIR = 'data'
OUTPUT_DIR = 'preprocessed_data'
CLIENTS = 3
TEST_SPLIT = 0.2

# Map class names to numeric labels
CLASS_MAP = {
    'Non Demented': 0,
    'Very mild Dementia': 1,
    'Mild Dementia': 2,
    'Moderate Dementia': 3
}

# Create output directories
os.makedirs(OUTPUT_DIR, exist_ok=True)
for i in range(CLIENTS):
    os.makedirs(os.path.join(OUTPUT_DIR, f'client_{i}'), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'test'), exist_ok=True)

# For each class, split images into clients and test set
for class_name, label in CLASS_MAP.items():
    class_dir = os.path.join(DATA_DIR, class_name)
    images = glob(os.path.join(class_dir, '*.jpg'))
    random.shuffle(images)
    n_total = len(images)
    n_test = int(TEST_SPLIT * n_total)
    test_images = images[:n_test]
    client_images = images[n_test:]
    print(f"Class '{class_name}' (label {label}): {n_total} images -> {n_test} test, {len(client_images)} for clients.")
    # Distribute to test set with progress bar
    for img_path in tqdm(test_images, desc=f"Copying test images for '{class_name}'"):
        base = os.path.basename(img_path)
        new_name = f"{os.path.splitext(base)[0]}label{label}.jpg"
        shutil.copy(img_path, os.path.join(OUTPUT_DIR, 'test', new_name))
    # Distribute to clients with progress bar
    for idx, img_path in enumerate(tqdm(client_images, desc=f"Copying client images for '{class_name}'")):
        client_id = idx % CLIENTS
        base = os.path.basename(img_path)
        new_name = f"{os.path.splitext(base)[0]}label{label}.jpg"
        shutil.copy(img_path, os.path.join(OUTPUT_DIR, f'client_{client_id}', new_name))
        if (idx+1) % 100 == 0 or (idx+1) == len(client_images):
            print(f"  Client {client_id}: {idx+1} / {len(client_images)} images distributed.")

print('Preprocessing complete. Data split into clients and test set.')