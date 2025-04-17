import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
from PIL import Image
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AlzheimersDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.images = []
        self.labels = []
        
        # Load all .jpg files and parse labels from filenames
        for file_name in os.listdir(data_dir):
            if file_name.lower().endswith('.jpg'):
                file_path = os.path.join(data_dir, file_name)
                # Extract label from filename (e.g., '_label_0')
                try:
                    label = int(file_name.split('_label_')[-1].replace('.jpg', ''))
                    self.images.append(file_path)
                    self.labels.append(label)
                except (ValueError, IndexError) as e:
                    logger.warning(f"Skipping {file_name}: Invalid label format ({e})")
        
        logger.info(f"Found {len(self.images)} valid images in {data_dir}")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        # Load and preprocess image
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        return image, label

def load_dataset(data_dir, batch_size=32, shuffle=True):
    """
    Load dataset from a directory and return a DataLoader.
    Assumes images are named with labels (e.g., 'train_No_Impairment_0_label_0.jpg').
    
    Args:
        data_dir (str): Path to the dataset directory (e.g., 'client_0' or 'test').
        batch_size (int): Batch size for the DataLoader.
        shuffle (bool): Whether to shuffle the dataset.
    
    Returns:
        DataLoader: PyTorch DataLoader for the dataset.
    """
    try:
        # Define transformations for ViT
        transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Already resized, but ensure consistency
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        dataset = AlzheimersDataset(data_dir, transform=transform)
        if len(dataset) == 0:
            raise ValueError(f"No valid images found in {data_dir}")
        
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=2,
            pin_memory=True
        )
        
        logger.info(f"Loaded DataLoader from {data_dir} with {len(dataset)} samples")
        return data_loader
    
    except Exception as e:
        logger.error(f"Error loading dataset from {data_dir}: {e}")
        raise
