import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
from PIL import Image
import logging
from dotenv import load_dotenv
import math

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AlzheimersDataset(Dataset):
    """Dataset for Alzheimer's images. Expects images in a flat directory (e.g., preprocessed_data/client_0 or preprocessed_data/test),
    with labels encoded in filenames as _label_{label}.jpg (e.g., *_label_0.jpg for Non Demented).
    """
    def __init__(self, data_dir, transform=None, partition=None, num_partitions=20):
        self.data_dir = data_dir
        self.transform = transform
        self.images = []
        self.labels = []
        all_files = []
        for file_name in os.listdir(data_dir):
            if file_name.lower().endswith('.jpg'):
                file_path = os.path.join(data_dir, file_name)
                try:
                    label = int(file_name.split('_label_')[-1].replace('.jpg', ''))
                    all_files.append((file_path, label))
                except (ValueError, IndexError) as e:
                    logger.warning(f"Skipping {file_name}: Invalid label format ({e})")
        # Partitioning logic
        if partition is not None and num_partitions > 1:
            total = len(all_files)
            part_size = math.ceil(total / num_partitions)
            start = (partition - 1) * part_size
            end = min(start + part_size, total)
            selected = all_files[start:end]
            logger.info(f"Partition {partition}/{num_partitions}: Using {len(selected)} of {total} samples from {data_dir}")
            self.images = [x[0] for x in selected]
            self.labels = [x[1] for x in selected]
        else:
            self.images = [x[0] for x in all_files]
            self.labels = [x[1] for x in all_files]
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

def load_dataset(data_dir, batch_size=32, shuffle=True, num_workers=4, augment=False, partition=None, num_partitions=20):
    """
    Load dataset and return a DataLoader.
    Args:
        data_dir (str): Path to dataset directory.
        batch_size (int): Batch size for DataLoader.
        shuffle (bool): Whether to shuffle the dataset.
        num_workers (int): Number of workers for data loading.
        augment (bool): Whether to use data augmentation (for training).
        partition (int): Which partition to use (1-based, for FL round).
        num_partitions (int): Total number of partitions (default 20).
    Returns:
        DataLoader: PyTorch DataLoader for the dataset.
    """
    try:
        if augment:
            transform = transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(10),
                transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
                transforms.RandomCrop(224, padding=8),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x + 0.05 * torch.randn_like(x)),  # Gaussian noise on tensor
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        dataset = AlzheimersDataset(data_dir, transform=transform, partition=partition, num_partitions=num_partitions)
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