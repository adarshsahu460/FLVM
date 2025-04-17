import h5py
import torch
import os
from fastapi import FastAPI, HTTPException, UploadFile, Depends
from fastapi.responses import FileResponse
from fastapi.security import APIKeyHeader
from models import ViTForAlzheimers
from utils import load_dataset
import logging
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI()

# API key authentication
VALID_API_KEYS = os.getenv("VALID_API_KEYS", "").split(",")
api_key_header = APIKeyHeader(name="X-API-Key")

# Directory to store client weights
WEIGHTS_DIR = os.getenv("WEIGHTS_DIR", "client_weights")
os.makedirs(WEIGHTS_DIR, exist_ok=True)

# Global model configuration
global_model = ViTForAlzheimers(num_labels=4)
GLOBAL_MODEL_PATH = os.getenv("GLOBAL_MODEL_PATH", "global_model.h5")
MODEL_VERSION = "1.0"  # Increment after each aggregation
VALIDATION_DATA_DIR = os.getenv("VALIDATION_DATA_DIR", "validation_data")

@app.post("/upload-weights/{client_id}")
async def upload_weights(client_id: str, file: UploadFile, dataset_size: int, api_key: str = Depends(api_key_header)):
    """Receive client weights and dataset size."""
    if api_key not in VALID_API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    try:
        if not file.filename.endswith('.h5'):
            raise HTTPException(status_code=400, detail="File must be in HDF5 format (.h5)")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        weight_path = os.path.join(WEIGHTS_DIR, f"weights_{client_id}_{timestamp}.h5")
        
        with open(weight_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Save dataset size for weighted aggregation
        metadata_path = weight_path.replace(".h5", ".txt")
        with open(metadata_path, "w") as f:
            f.write(str(dataset_size))
        
        logger.info(f"Received weights and dataset size {dataset_size} from client {client_id}, saved to {weight_path}")
        return {"message": f"Weights from client {client_id} saved successfully"}
    
    except Exception as e:
        logger.error(f"Error receiving weights from client {client_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Error saving weights: {str(e)}")

@app.get("/get-global-model")
async def get_global_model(api_key: str = Depends(api_key_header)):
    """Serve the global model with version header."""
    if api_key not in VALID_API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    try:
        if not os.path.exists(GLOBAL_MODEL_PATH):
            with h5py.File(GLOBAL_MODEL_PATH, 'w') as f:
                for key, param in global_model.state_dict().items():
                    f.create_dataset(key, data=param.cpu().numpy())
            logger.info("Initialized and saved global model")
        
        return FileResponse(
            path=GLOBAL_MODEL_PATH,
            media_type="application/x-hdf5",
            filename="global_model.h5",
            headers={"X-Model-Version": MODEL_VERSION}
        )
    
    except Exception as e:
        logger.error(f"Error serving global model: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving global model: {str(e)}")

@app.get("/validate-global-model")
async def validate_global_model():
    """Evaluate global model on validation dataset."""
    try:
        if not os.path.exists(GLOBAL_MODEL_PATH):
            raise HTTPException(status_code=404, detail="Global model not found")
        
        model = ViTForAlzheimers(num_labels=4)
        with h5py.File(GLOBAL_MODEL_PATH, 'r') as f:
            state_dict = {key: torch.tensor(np.array(f[key])) for key in f.keys()}
            model.load_state_dict(state_dict, strict=True)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        
        validation_loader = load_dataset(VALIDATION_DATA_DIR, shuffle=False)
        criterion = torch.nn.CrossEntropyLoss()
        total_loss, correct, total = 0, 0, 0
        
        with torch.no_grad():
            for images, labels in validation_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        avg_loss = total_loss / len(validation_loader)
        accuracy = 100 * correct / total
        logger.info(f"Validation Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
        return {"loss": avg_loss, "accuracy": accuracy}
    
    except Exception as e:
        logger.error(f"Error validating global model: {e}")
        raise HTTPException(status_code=500, detail=f"Error validating global model: {str(e)}")