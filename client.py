import h5py
import torch
import torch.optim as optim
from torch.nn import CrossEntropyLoss
from transformers import ViTModel
import logging
import sys
import requests
from utils import load_dataset
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ViTForAlzheimers(torch.nn.Module):
    def __init__(self, num_labels=4):
        super(ViTForAlzheimers, self).__init__()
        self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(self.vit.config.hidden_size, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(256, num_labels)
        )
    
    def forward(self, pixel_values):
        outputs = self.vit(pixel_values=pixel_values)
        pooled_output = outputs.last_hidden_state[:, 0]
        logits = self.classifier(pooled_output)
        return logits

class AlzheimersClient:
    def __init__(self, model, train_loader, cid, server_url="http://localhost:8080"):
        self.model = model
        self.train_loader = train_loader
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.7)
        self.criterion = CrossEntropyLoss()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.cid = cid
        self.server_url = server_url
    
    def load_global_model(self):
        logger.info(f"Client {self.cid}: Downloading global model")
        try:
            response = requests.get(f"{self.server_url}/get-global-model")
            response.raise_for_status()
            # Write the downloaded HDF5 file to disk
            with open("temp_global_model.h5", 'wb') as f:
                f.write(response.content)
            # Read the HDF5 file to load model weights
            with h5py.File("temp_global_model.h5", 'r') as f:
                state_dict = {key: torch.tensor(np.array(f[key])) for key in f.keys()}
                self.model.load_state_dict(state_dict, strict=True)
            logger.info(f"Client {self.cid}: Loaded global model")
        except Exception as e:
            logger.error(f"Client {self.cid}: Error loading global model: {e}")
            raise

    def train(self):
        logger.info(f"Client {self.cid}: Starting training")
        self.model.train()
        
        total_batches_processed = 0
        max_batches = 100
        num_epochs = 2
        
        for epoch in range(num_epochs):
            if total_batches_processed >= max_batches:
                logger.info(f"Client {self.cid}: Reached maximum of {max_batches} batches")
                break
                
            logger.info(f"Client {self.cid}: Epoch {epoch + 1}/{num_epochs}")
            batch_count = 0
            for batch_idx, (images, labels) in enumerate(self.train_loader):
                if total_batches_processed >= max_batches:
                    logger.info(f"Client {self.cid}: Reached maximum of {max_batches} batches in epoch {epoch + 1}")
                    break
                    
                batch_count += 1
                logger.info(f"Running batch: {batch_count}")
                total_batches_processed += 1
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                logger.debug(f"Client {self.cid}: Batch {batch_idx + 1} loss: {loss.item():.4f}")
            
            self.scheduler.step()
            logger.info(f"Client {self.cid}: Completed epoch {epoch + 1} with {batch_count} batches")
        
        logger.info(f"Client {self.cid}: Finished training with {total_batches_processed} total batches")
    
    def save_and_send_weights(self):
        logger.info(f"Client {self.cid}: Saving and sending weights")
        weight_path = f"client_{self.cid}_weights.h5"
        try:
            with h5py.File(weight_path, 'w') as f:
                for key, param in self.model.state_dict().items():
                    f.create_dataset(key, data=param.cpu().numpy())
            
            with open(weight_path, 'rb') as f:
                response = requests.post(
                    f"{self.server_url}/upload-weights/{self.cid}",
                    files={"file": (weight_path, f, "application/x-hdf5")}
                )
                response.raise_for_status()
            logger.info(f"Client {self.cid}: Weights sent to server")
        except Exception as e:
            logger.error(f"Client {self.cid}: Error sending weights: {e}")
            raise

def start_client(cid):
    logger.info(f"Starting client {cid}")
    try:
        model = ViTForAlzheimers()
        train_loader = load_dataset(f"/home/adarsh/Projects/FLVM_MP_PyTorch/preprocessed_data/client_{cid}")
        client = AlzheimersClient(model, train_loader, cid)
        client.load_global_model()
        client.train()
        client.save_and_send_weights()
    except Exception as e:
        logger.error(f"Client {cid}: Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    cid = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    start_client(cid)
