import h5py
import torch
import numpy as np
from collections import OrderedDict
from models import ViTForAlzheimers
import logging
import os
from multiprocessing import Pool
from functools import partial
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

SERVER_MOMENTUM = float(os.getenv("SERVER_MOMENTUM", 0.7))  # 70% old, 30% new by default

def process_weights(weight_file, weights_dir):
    """Load weights and dataset size for a client."""
    try:
        weight_path = os.path.join(weights_dir, weight_file)
        metadata_path = weight_path.replace(".h5", ".txt")
        with h5py.File(weight_path, 'r') as f:
            state_dict = {key: torch.tensor(np.array(f[key])) for key in f.keys()}
        with open(metadata_path, "r") as f:
            dataset_size = int(f.read())
        return weight_file, state_dict, dataset_size
    except Exception as e:
        logger.error(f"Error processing {weight_file}: {e}")
        return weight_file, None, 0

def aggregate_weights():
    """Aggregate client weights using weighted FedAvg."""
    WEIGHTS_DIR = os.getenv("WEIGHTS_DIR", "client_weights")
    GLOBAL_MODEL_PATH = os.getenv("GLOBAL_MODEL_PATH", "global_model.h5")
    model = ViTForAlzheimers(num_labels=4)
    # Get latest weights per client
    client_weights = {}
    for f in os.listdir(WEIGHTS_DIR):
        if f.endswith('.h5'):
            client_id = f.split('_')[1]
            timestamp = f.split('_')[-1].replace('.h5', '')
            if client_id not in client_weights or timestamp > client_weights[client_id][1]:
                client_weights[client_id] = (f, timestamp)
    weight_files = [info[0] for info in client_weights.values()]
    if not weight_files:
        logger.warning("No client weights found for aggregation")
        return
    # Load weights in parallel
    logger.info(f"Found {len(weight_files)} client weights for aggregation")
    with Pool() as pool:
        results = list(tqdm(pool.imap(partial(process_weights, weights_dir=WEIGHTS_DIR), weight_files), total=len(weight_files), desc="Processing client weights"))
    # Aggregate weights
    aggregated_state_dict = OrderedDict()
    total_samples = 0
    global_shapes = {k: v.shape for k, v in model.state_dict().items()}
    model_keys = set(model.state_dict().keys())
    for weight_file, client_state_dict, dataset_size in results:
        if client_state_dict is None:
            continue
        # Check for missing/unexpected keys
        client_keys = set(client_state_dict.keys())
        missing_keys = model_keys - client_keys
        unexpected_keys = client_keys - model_keys
        if missing_keys:
            logger.error(f"Missing keys in {weight_file}: {missing_keys}")
        if unexpected_keys:
            logger.error(f"Unexpected keys in {weight_file}: {unexpected_keys}")
        try:
            # Validate shapes
            for key in client_state_dict:
                if key in global_shapes and client_state_dict[key].shape != global_shapes[key]:
                    logger.error(f"Invalid shape for {key} in {weight_file}")
                    continue
            if not aggregated_state_dict:
                for key in client_state_dict:
                    if key in model_keys:
                        aggregated_state_dict[key] = client_state_dict[key] * dataset_size
            else:
                for key in client_state_dict:
                    if key in model_keys:
                        aggregated_state_dict[key] += client_state_dict[key] * dataset_size
            total_samples += dataset_size
            logger.info(f"Processed weights from {weight_file} with {dataset_size} samples")
        except Exception as e:
            logger.error(f"Error processing {weight_file}: {e}")
            continue
    # Normalize aggregated weights
    if total_samples > 0:
        for key in tqdm(aggregated_state_dict, desc="Normalizing aggregated weights"):
            aggregated_state_dict[key] /= total_samples
    # Blend with previous global model weights (server-side momentum)
    if os.path.exists(GLOBAL_MODEL_PATH):
        with h5py.File(GLOBAL_MODEL_PATH, 'r') as f:
            old_global_state = {key: torch.tensor(np.array(f[key])) for key in f.keys()}
        for key in aggregated_state_dict:
            if key in old_global_state:
                aggregated_state_dict[key] = (
                    SERVER_MOMENTUM * old_global_state[key] +
                    (1 - SERVER_MOMENTUM) * aggregated_state_dict[key]
                )
    # Update global model
    try:
        # Log missing/unexpected keys on load
        load_result = model.load_state_dict(aggregated_state_dict, strict=False)
        if load_result.missing_keys:
            logger.error(f"Missing keys when loading aggregated state dict: {load_result.missing_keys}")
        if load_result.unexpected_keys:
            logger.error(f"Unexpected keys when loading aggregated state dict: {load_result.unexpected_keys}")
        with h5py.File(GLOBAL_MODEL_PATH, 'w') as f:
            for key, param in model.state_dict().items():
                f.create_dataset(key, data=param.cpu().numpy())
        logger.info(f"Aggregated and saved global model to {GLOBAL_MODEL_PATH}")
        # Archive processed weight files
        archive_dir = os.path.join(WEIGHTS_DIR, "archive")
        os.makedirs(archive_dir, exist_ok=True)
        for weight_file, _, _ in results:
            if weight_file:
                os.rename(
                    os.path.join(WEIGHTS_DIR, weight_file),
                    os.path.join(archive_dir, weight_file)
                )
                metadata_path = weight_file.replace(".h5", ".txt")
                if os.path.exists(os.path.join(WEIGHTS_DIR, metadata_path)):
                    os.rename(
                        os.path.join(WEIGHTS_DIR, metadata_path),
                        os.path.join(archive_dir, metadata_path)
                    )
        logger.info("Archived processed weight files")
    except Exception as e:
        logger.error(f"Error saving aggregated model: {e}")

if __name__ == "__main__":
    logger.info("Starting nightly aggregation...")
    aggregate_weights()