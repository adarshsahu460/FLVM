import os
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AlzheimersDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.files = [f for f in os.listdir(data_dir) if f.endswith('.jpg')]
        if not self.files:
            raise ValueError(f"No .jpg files found in {data_dir}")
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
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
            logger.error(f"Error loading image {self.files[idx]}: {e}")
            raise

def load_dataset(data_dir, batch_size=4, shuffle=True):
    dataset = AlzheimersDataset(data_dir)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)