import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
from PIL import Image
import logging
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AlzheimersDataset(Dataset):
    """Dataset for Alzheimer's images with labels extracted from filenames."""
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.images = []
        self.labels = []
        
        for file_name in os.listdir(data_dir):
            if file_name.lower().endswith('.jpg'):
                file_path = os.path.join(data_dir, file_name)
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
        
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        return image, label

def load_dataset(data_dir, batch_size=32, shuffle=True, num_workers=4):
    """
    Load dataset and return a DataLoader.
    
    Args:
        data_dir (str): Path to dataset directory.
        batch_size (int): Batch size for DataLoader.
        shuffle (bool): Whether to shuffle the dataset.
        num_workers (int): Number of workers for data loading.
    
    Returns:
        DataLoader: PyTorch DataLoader for the dataset.
    """
    try:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
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
            num_workers=num_workers if torch.cuda.is_available() else 0,
            pin_memory=torch.cuda.is_available()
        )
        
        logger.info(f"Loaded DataLoader from {data_dir} with {len(dataset)} samples")
        return data_loader
    
    except Exception as e:
        logger.error(f"Error loading dataset from {data_dir}: {e}")
        raise