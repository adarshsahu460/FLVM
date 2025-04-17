import h5py
import torch
import numpy as np
from collections import OrderedDict
from transformers import ViTModel
import logging
import os
from datetime import datetime

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

def aggregate_weights():
    WEIGHTS_DIR = "client_weights"
    GLOBAL_MODEL_PATH = "global_model.h5"
    model = ViTForAlzheimers(num_labels=4)
    
    # Get all weight files from today
    today = datetime.now().strftime("%Y%m%d")
    weight_files = [f for f in os.listdir(WEIGHTS_DIR) if f.endswith('.h5') and today in f]
    
    if not weight_files:
        logger.warning("No client weights found for aggregation")
        return
    
    # Load and average weights
    logger.info(f"Found {len(weight_files)} client weights for aggregation")
    aggregated_state_dict = OrderedDict()
    num_clients = len(weight_files)
    
    for idx, weight_file in enumerate(weight_files):
        weight_path = os.path.join(WEIGHTS_DIR, weight_file)
        try:
            with h5py.File(weight_path, 'r') as f:
                client_state_dict = {key: torch.tensor(np.array(f[key])) for key in f.keys()}
                
                # Initialize aggregated state dict on first client
                if idx == 0:
                    for key in client_state_dict:
                        aggregated_state_dict[key] = client_state_dict[key] / num_clients
                else:
                    for key in client_state_dict:
                        aggregated_state_dict[key] += client_state_dict[key] / num_clients
                        
            logger.info(f"Processed weights from {weight_file}")
        except Exception as e:
            logger.error(f"Error processing {weight_file}: {e}")
            continue
    
    # Update global model
    try:
        model.load_state_dict(aggregated_state_dict, strict=True)
        with h5py.File(GLOBAL_MODEL_PATH, 'w') as f:
            for key, param in model.state_dict().items():
                f.create_dataset(key, data=param.cpu().numpy())
        logger.info(f"Aggregated and saved global model to {GLOBAL_MODEL_PATH}")
        
        # Optionally, clean up processed weight files
        for weight_file in weight_files:
            os.remove(os.path.join(WEIGHTS_DIR, weight_file))
        logger.info("Cleaned up processed weight files")
        
    except Exception as e:
        logger.error(f"Error saving aggregated model: {e}")

if __name__ == "__main__":
    logger.info("Starting nightly aggregation...")
    aggregate_weights()