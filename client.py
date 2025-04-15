import flwr as fl
import torch
import torch.optim as optim
from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
from transformers import ViTModel
import os
import sys

class ViTForAlzheimers(torch.nn.Module):
    def __init__(self, num_labels=4):
        super(ViTForAlzheimers, self).__init__()
        self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
        self.classifier = torch.nn.Linear(self.vit.config.hidden_size, num_labels)
    
    def forward(self, pixel_values):
        outputs = self.vit(pixel_values=pixel_values)
        pooled_output = outputs.last_hidden_state[:, 0]
        logits = self.classifier(pooled_output)
        return logits

class AlzheimersDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        try:
            self.files = [f for f in os.listdir(data_dir) if f.endswith('.jpg')]
            print(f"Found {len(self.files)} files in {data_dir}")
            if not self.files:
                raise ValueError(f"No .jpg files found in {data_dir}")
        except Exception as e:
            print(f"Error loading dataset: {e}")
            raise
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
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
            print(f"Error loading image {self.files[idx]}: {e}")
            raise

class AlzheimersClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, test_loader, cid):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = CrossEntropyLoss()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.cid = cid
        self.has_trained = False  # Track if training has occurred

    def get_parameters(self, config):
        print(f"Client {self.cid}: Getting parameters")
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
    
    def set_parameters(self, parameters):
        print(f"Client {self.cid}: Setting parameters")
        try:
            params_dict = zip(self.model.state_dict().keys(), parameters)
            state_dict = {k: torch.tensor(v) for k, v in params_dict}
            self.model.load_state_dict(state_dict, strict=True)
        except Exception as e:
            print(f"Client {self.cid}: Error setting parameters: {e}")
            raise
    
    def fit(self, parameters, config):
        if self.has_trained:
            print(f"Client {self.cid}: Already trained, skipping further training")
            sys.exit(0)  # Exit if called again
        print(f"Client {self.cid}: Starting training")
        self.set_parameters(parameters)
        self.model.train()
        try:
            batch_count = 0
            max_batches = 10  # Limit to 10 batches
            for i, (images, labels) in enumerate(self.train_loader):
                if batch_count >= max_batches:
                    print(f"Client {self.cid}: Reached 10 batches, stopping training")
                    break
                print(f"Client {self.cid}: Processing batch {i+1}/{max_batches}")
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                batch_count += 1
            print(f"Client {self.cid}: Finished training on {batch_count} batches")
            self.has_trained = True  # Mark as trained
        except Exception as e:
            print(f"Client {self.cid}: Training error: {e}")
            raise
        print(f"Client {self.cid}: Sending weights")
        return self.get_parameters(config), len(self.train_loader.dataset), {"terminate": True}

    def evaluate(self, parameters, config):
        print(f"Client {self.cid}: Evaluate called (unused)")
        return float(0), 0, {"accuracy": 0.0}

def start_client(cid):
    print(f"Starting client {cid}")
    try:
        model = ViTForAlzheimers()
        train_loader = DataLoader(
            AlzheimersDataset(f"/home/adarsh/Projects/FLVM_MP_PyTorch/preprocessed_data/client_{cid}"),
            batch_size=4,
            shuffle=True
        )
        test_loader = DataLoader(
            AlzheimersDataset("/home/adarsh/Projects/FLVM_MP_PyTorch/preprocessed_data/test"),
            batch_size=4,
            shuffle=False
        )
        client = AlzheimersClient(model, train_loader, test_loader, cid)
        print(f"Client {cid}: Connecting to server")
        fl.client.start_numpy_client(
            server_address="localhost:8080",
            client=client
        )
    except Exception as e:
        print(f"Client {cid}: Error: {e}")
    finally:
        print(f"Client {cid}: Sent weights and terminating")
        sys.exit(0)

if __name__ == "__main__":
    cid = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    start_client(cid)