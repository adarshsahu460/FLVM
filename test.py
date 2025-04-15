import h5py
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
from transformers import ViTModel
import os
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Define ViTForAlzheimers class (same as server.py and client.py)
class ViTForAlzheimers(torch.nn.Module):
    def __init__(self, num_labels=4):
        super(ViTForAlzheimers, self).__init__()
        self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
        self.classifier = torch.nn.Linear(self.vit.config.hidden_size, num_labels)
    
    def forward(self, pixel_values):
        outputs = self.vit(pixel_values=pixel_values)
        pooled_output = outputs.last_hidden_state[:, 0]
        logits = self.classifier(pooled_output)
        return logits

# Define AlzheimersDataset class (same as client.py)
class AlzheimersDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        try:
            self.files = [f for f in os.listdir(data_dir) if f.endswith('.jpg')]
            print(f"Found {len(self.files)} files in {data_dir}")
            if not self.files:
                raise ValueError(f"No .jpg files found in {data_dir}")
        except Exception as e:
            print(f"Error loading dataset: {e}")
            raise
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.485, 0.485], std=[0.229, 0.229, 0.229])
        ])
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        try:
            img_path = os.path.join(self.data_dir, self.files[idx])
            label = int(self.files[idx].split('_label_')[1].split('.jpg')[0])
            img = Image.open(img_path).convert('RGB')
            img = self.transform(img)
            return img, label
        except Exception as e:
            print(f"Error loading image {self.files[idx]}: {e}")
            raise

# Function to evaluate model accuracy
def evaluate_model(model_path, test_loader, device):
    print(f"Evaluating model from {model_path}")
    try:
        # Initialize model
        model = ViTForAlzheimers(num_labels=4)
        model.to(device)
        model.eval()

        # Load weights from model.h5
        with h5py.File(model_path, 'r') as f:
            state_dict = {k: torch.tensor(v[:]).to(device) for k, v in f.items()}
            model.load_state_dict(state_dict)
        print("Model weights loaded successfully")

        # Evaluate on test dataset
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        print(f"Accuracy on test dataset: {accuracy:.2f}% ({correct}/{total})")
        return accuracy
    except Exception as e:
        print(f"Error during evaluation: {e}")
        return None

# File watcher to detect model.h5 updates
class ModelFileHandler(FileSystemEventHandler):
    def __init__(self, model_path, test_loader, device):
        self.model_path = model_path
        self.test_loader = test_loader
        self.device = device
        self.last_modified = 0  # Track last modification time to avoid duplicates

    def on_modified(self, event):
        if event.src_path == self.model_path:
            # Check file modification time to avoid duplicate triggers
            current_modified = os.path.getmtime(self.model_path)
            if current_modified > self.last_modified:
                self.last_modified = current_modified
                print(f"Detected update to {self.model_path}")
                evaluate_model(self.model_path, self.test_loader, self.device)

def main():
    model_path = "model.h5"
    test_data_dir = "/home/adarsh/Projects/FLVM_MP_PyTorch/preprocessed_data/test"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize test dataset
    try:
        test_dataset = AlzheimersDataset(test_data_dir)
        test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)
    except Exception as e:
        print(f"Failed to load test dataset: {e}")
        return

    # Check if model.h5 exists initially
    if os.path.exists(model_path):
        evaluate_model(model_path, test_loader, device)
    else:
        print(f"{model_path} not found, waiting for first update")

    # Set up file watcher
    event_handler = ModelFileHandler(model_path, test_loader, device)
    observer = Observer()
    observer.schedule(event_handler, path=".", recursive=False)
    observer.start()
    print(f"Watching {model_path} for updates...")

    try:
        while True:
            time.sleep(1)  # Keep watcher running
    except KeyboardInterrupt:
        print("Stopping file watcher")
        observer.stop()
    observer.join()

if __name__ == "__main__":
    main()