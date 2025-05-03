import h5py
import torch
import torch.optim as optim
from torch.nn import CrossEntropyLoss
from torch.cuda.amp import autocast, GradScaler
import logging
import sys
import requests
from utils import load_dataset
from models import ViTForAlzheimers
import numpy as np
from dotenv import load_dotenv
import os
import time
from tqdm import tqdm

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AlzheimersClient:
    def _init_(self, model, train_loader, cid, server_url, api_key):
        self.model = model
        self.train_loader = train_loader
        self.optimizer = optim.Adam(self.model.parameters(), lr=float(os.getenv("LEARNING_RATE", 0.0001)))
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.7)
        self.criterion = CrossEntropyLoss()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.cid = cid
        self.server_url = server_url
        self.api_key = api_key
        self.scaler = GradScaler() if torch.cuda.is_available() else None
    
    def load_global_model(self):
        """Download and load the global model, verifying version."""
        logger.info(f"Client {self.cid}: Downloading global model")
        headers = {"X-API-Key": self.api_key}
        retries = 5
        for attempt in range(retries):
            try:
                response = requests.get(f"{self.server_url}/get-global-model", headers=headers)
                response.raise_for_status()
                expected_version = os.getenv("MODEL_VERSION", "1.0")
                if response.headers.get("X-Model-Version") != expected_version:
                    raise ValueError(f"Model version mismatch: expected {expected_version}, got {response.headers.get('X-Model-Version')}")
                with open("temp_global_model.h5", 'wb') as f:
                    f.write(response.content)
                with h5py.File("temp_global_model.h5", 'r') as f:
                    state_dict = {key: torch.tensor(np.array(f[key])) for key in f.keys()}
                    self.model.load_state_dict(state_dict, strict=True)
                logger.info(f"Client {self.cid}: Loaded global model")
                return
            except Exception as e:
                logger.error(f"Client {self.cid}: Error loading global model (attempt {attempt+1}/{retries}): {e}")
                time.sleep(1 + self.cid)  # Stagger by client id
        raise RuntimeError(f"Client {self.cid}: Failed to load global model after {retries} attempts")
    
    def add_dp_noise(self, state_dict, noise_scale=0.1):
        """Add Gaussian noise to weights for differential privacy."""
        for key in state_dict:
            noise = torch.normal(0, noise_scale, size=state_dict[key].size()).to(state_dict[key].device)
            state_dict[key] += noise
        return state_dict
    
    def train(self):
        """Train the model with mixed precision."""
        logger.info(f"Client {self.cid}: Starting training")
        self.model.train()
        
        total_batches_processed = 0
        # max_batches = int(os.getenv("MAX_BATCHES", 100))
        num_epochs = int(os.getenv("NUM_EPOCHS", 20))
        
        for epoch in range(num_epochs):
            # if total_batches_processed >= max_batches:
                # logger.info(f"Client {self.cid}: Reached maximum of {max_batches} batches")
                # break
            
            logger.info(f"Client {self.cid}: Epoch {epoch + 1}/{num_epochs}")
            batch_count = 0
            for batch_idx, (images, labels) in enumerate(tqdm(self.train_loader, desc=f"Client {self.cid} Epoch {epoch+1}/{num_epochs}")):
                # if total_batches_processed >= max_batches:
                #     logger.info(f"Client {self.cid}: Reached maximum of {max_batches} batches in epoch {epoch + 1}")
                #     break
                
                batch_count += 1
                # total_batches_processed += 1
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                
                if self.scaler:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    self.optimizer.step()
                
                logger.debug(f"Client {self.cid}: Batch {batch_idx + 1} loss: {loss.item():.4f}")
            
            self.scheduler.step()
            logger.info(f"Client {self.cid}: Completed epoch {epoch + 1} with {batch_count} batches")
        
        logger.info(f"Client {self.cid}: Finished training with {total_batches_processed} total batches")
    
    def save_and_send_weights(self):
        """Save and send weights with dataset size."""
        logger.info(f"Client {self.cid}: Saving and sending weights")
        weight_path = f"client_{self.cid}_weights.h5"
        try:
            state_dict = self.add_dp_noise(self.model.state_dict(), noise_scale=float(os.getenv("DP_NOISE_SCALE", 0.1)))
            with h5py.File(weight_path, 'w') as f:
                for key, param in state_dict.items():
                    f.create_dataset(key, data=param.cpu().numpy())
            
            dataset_size = len(self.train_loader.dataset)
            with open(weight_path, 'rb') as f:
                response = requests.post(
                    f"{self.server_url}/upload-weights/{self.cid}",
                    files={"file": (weight_path, f, "application/x-hdf5")},
                    data={"dataset_size": dataset_size},
                    headers={"X-API-Key": self.api_key}
                )
                response.raise_for_status()
            logger.info(f"Client {self.cid}: Weights sent to server")
        except Exception as e:
            logger.error(f"Client {self.cid}: Error sending weights: {e}")
            raise

def start_client(cid, data_dir, server_url, api_key, round_num=1):
    """Start the client with given configuration."""
    logger.info(f"Starting client {cid} for round {round_num}")
    try:
        model = ViTForAlzheimers()
        # Use entire dataset per round, batch size 8 for low memory
        train_loader = load_dataset(data_dir, batch_size=8, augment=True, partition=None)
        client = AlzheimersClient(model, train_loader, cid, server_url, api_key)
        client.load_global_model()
        client.train()
        client.save_and_send_weights()
    except Exception as e:
        logger.error(f"Client {cid}: Error: {e}")
        sys.exit(1)

if __name__ == "_main_":
    import argparse
    parser = argparse.ArgumentParser(description="Federated Learning Client")
    parser.add_argument("--cid", type=int, default=0, help="Client ID")
    parser.add_argument("--data-dir", default=os.getenv("DATA_DIR", "preprocessed_data"), help="Dataset directory")
    parser.add_argument("--round", type=int, default=1, help="Federated round number (1-based)")
    args = parser.parse_args()
    
    server_url = os.getenv("SERVER_URL", "http://localhost:8080")
    api_key = os.getenv("API_KEY")
    if not api_key:
        logger.error("API_KEY not set in .env")
        sys.exit(1)
    if not os.getenv("DP_NOISE_SCALE"):
        os.environ["DP_NOISE_SCALE"] = "0.0"
    start_client(args.cid, os.path.join(args.data_dir, f"client_{args.cid}"), server_url, api_key, round_num=args.round)