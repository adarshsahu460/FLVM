import os
import shutil
from PIL import Image
import numpy as np
import time

def preprocess_dataset(input_dir, output_dir, num_clients=5):
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    # Define class labels
    classes = ["No Impairment", "Very Mild Impairment", "Mild Impairment", "Moderate Impairment"]
    label_map = {cls: idx for idx, cls in enumerate(classes)}
    print(f"Class labels: {label_map}")
    
    # Process train and test sets
    for split in ["train", "test"]:
        split_input_dir = os.path.join(input_dir, split)
        split_output_dir = os.path.join(output_dir, split)
        os.makedirs(split_output_dir, exist_ok=True)
        print(f"Processing split: {split}")
        print(f"Input split dir: {split_input_dir}")
        print(f"Output split dir: {split_output_dir}")
        
        all_images = []
        for cls in classes:
            cls_dir = os.path.join(split_input_dir, cls)
            print(f"Checking class directory: {cls_dir}")
            if not os.path.exists(cls_dir):
                print(f"Directory does not exist: {cls_dir}")
                continue
            files = [f for f in os.listdir(cls_dir) if f.lower().endswith('.jpg')]
            print(f"Found {len(files)} .jpg files in {cls_dir}")
            for img_file in files:
                all_images.append((cls, img_file))
        
        print(f"Total images found for {split}: {len(all_images)}")
        if not all_images:
            print(f"No images found in {split_input_dir}. Skipping.")
            continue
        
        # Process and save images
        for cls, img_file in all_images:
            img_path = os.path.join(split_input_dir, cls, img_file)
            print(f"Processing image: {img_path}")
            try:
                img = Image.open(img_path)
                img = img.resize((224, 224))
                label = label_map[cls]
                output_path = os.path.join(split_output_dir, f"{split}_{cls.replace(' ', '_')}_{len(os.listdir(split_output_dir))}_label_{label}.jpg")
                img.convert('RGB').save(output_path)
                print(f"Saved to: {output_path}")
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
    
    # Simulate federated clients for training data
    train_dir = os.path.join(output_dir, "train")
    train_files = [f for f in os.listdir(train_dir) if f.endswith('.jpg')]
    print(f"Train files for splitting: {len(train_files)}")
    if not train_files:
        print("No train files to split into clients.")
        return
    
    client_data = np.array_split(train_files, num_clients)
    for cid, files in enumerate(client_data):
        client_dir = os.path.join(output_dir, f"client_{cid}")
        os.makedirs(client_dir, exist_ok=True)
        print(f"Creating client_{cid} with {len(files)} files")
        for f in files:
            shutil.move(os.path.join(train_dir, f), os.path.join(client_dir, f))
            print(f"Moved {f} to {client_dir}")

if __name__ == "__main__":
    input_dir = "D:\Projects\FLVM\Combined_Dataset"
    output_dir = "D:\Projects\FLVM\preprocessed_data"
    preprocess_dataset(input_dir, output_dir)