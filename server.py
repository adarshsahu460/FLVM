import h5py
import torch
import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from collections import OrderedDict
from transformers import ViTModel
import logging
import os
from datetime import datetime
import uvicorn

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI()

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

# Directory to store client weights
WEIGHTS_DIR = "client_weights"
os.makedirs(WEIGHTS_DIR, exist_ok=True)

# Initialize global model
global_model = ViTForAlzheimers(num_labels=4)
GLOBAL_MODEL_PATH = "global_model.h5"

@app.post("/upload-weights/{client_id}")
async def upload_weights(client_id: str, file: UploadFile):
    try:
        # Validate file is HDF5
        if not file.filename.endswith('.h5'):
            raise HTTPException(status_code=400, detail="File must be in HDF5 format (.h5)")
        
        # Save weights with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        weight_path = os.path.join(WEIGHTS_DIR, f"weights_{client_id}_{timestamp}.h5")
        
        with open(weight_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        logger.info(f"Received weights from client {client_id}, saved to {weight_path}")
        return {"message": f"Weights from client {client_id} saved successfully"}
    
    except Exception as e:
        logger.error(f"Error receiving weights from client {client_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Error saving weights: {str(e)}")

@app.get("/get-global-model")
async def get_global_model():
    try:
        if not os.path.exists(GLOBAL_MODEL_PATH):
            # Save initial global model if it doesn't exist
            with h5py.File(GLOBAL_MODEL_PATH, 'w') as f:
                for key, param in global_model.state_dict().items():
                    f.create_dataset(key, data=param.cpu().numpy())
            logger.info("Initialized and saved global model")
        
        # Return the HDF5 file as a binary response
        return FileResponse(
            path=GLOBAL_MODEL_PATH,
            media_type="application/x-hdf5",
            filename="global_model.h5"
        )
    
    except Exception as e:
        logger.error(f"Error serving global model: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving global model: {str(e)}")

if __name__ == "__main__":
    logger.info("Starting FastAPI server...")
    uvicorn.run(app, host="0.0.0.0", port=8080)
