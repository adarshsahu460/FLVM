import flwr as fl
import torch
import torch.optim as optim
from torch.nn import CrossEntropyLoss
from transformers import ViTModel
import sys
import logging
from utils import load_dataset

# Configure logging
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

class AlzheimersClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, test_loader, cid):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.7)
        self.criterion = CrossEntropyLoss()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.cid = cid

    def get_parameters(self, config):
        logger.info(f"Client {self.cid}: Getting parameters")
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
    
    def set_parameters(self, parameters):
        logger.info(f"Client {self.cid}: Setting parameters")
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)
    
    def fit(self, parameters, config):
        server_round = config.get("server_round", 1)  # Get round number from server config
        num_epochs = min(config.get("num_epochs", 2), 10)  # Cap at 10 epochs
        dataset_path = self.train_loader.dataset.data_dir  # Get dataset path
        logger.info(f"Client {self.cid}: Starting training for round {server_round} on dataset {dataset_path}")

        self.set_parameters(parameters)
        self.model.train()
        
        total_batches_processed = 0  # Track total batches across epochs
        max_batches = 100  # Maximum batches to process
        max_epochs = 10  # Maximum epochs to run
        
        for epoch in range(num_epochs):
            if total_batches_processed >= max_batches:
                logger.info(f"Client {self.cid}: Reached maximum of {max_batches} batches, stopping training")
                break
                
            logger.info(f"Client {self.cid}: Round {server_round}, Epoch {epoch + 1}/{num_epochs}")
            batch_count = 0
            for batch_idx, (images, labels) in enumerate(self.train_loader):
                if total_batches_processed >= max_batches:
                    logger.info(f"Client {self.cid}: Reached maximum of {max_batches} batches in epoch {epoch + 1}")
                    break
                    
                batch_count += 1
                total_batches_processed += 1
                logger.info(f"Client {self.cid}: Processing batch {batch_idx + 1} in epoch {epoch + 1}")
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                logger.debug(f"Client {self.cid}: Batch {batch_idx + 1} loss: {loss.item():.4f}")
            
            self.scheduler.step()
            logger.info(f"Client {self.cid}: Completed epoch {epoch + 1} with {batch_count} batches")
        
        logger.info(f"Client {self.cid}: Finished training for round {server_round} with {total_batches_processed} total batches")
        logger.info(f"Client {self.cid}: Sending updated weights to server")
        return self.get_parameters(config), len(self.train_loader.dataset), {}
        
    def evaluate(self, parameters, config):
        logger.info(f"Client {self.cid}: Evaluate called (unused)")
        return float(0), 0, {"accuracy": 0.0}

def start_client(cid):
    logger.info(f"Starting client {cid}")
    try:
        model = ViTForAlzheimers()
        train_loader = load_dataset(f"/home/adarsh/Projects/FLVM_MP_PyTorch/preprocessed_data/client_{cid}")
        test_loader = load_dataset("/home/adarsh/Projects/FLVM_MP_PyTorch/preprocessed_data/test", shuffle=False)
        client = AlzheimersClient(model, train_loader, test_loader, cid)
        logger.info(f"Client {cid}: Connecting to server at localhost:8080")
        fl.client.start_numpy_client(
            server_address="localhost:8080",
            client=client
        )
    except Exception as e:
        logger.error(f"Client {cid}: Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    cid = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    start_client(cid)