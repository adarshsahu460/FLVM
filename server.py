import flwr as fl
import torch
import h5py
from collections import OrderedDict
from transformers import ViTModel
import logging

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

# Custom strategy to process one client at a time and save model
class SingleClientUpdateStrategy(fl.server.strategy.Strategy):
    def __init__(self, model, save_path="model.h5"):
        super().__init__()
        self.model = model
        self.save_path = save_path

    def initialize_parameters(self, client_manager):
        print("Initializing parameters")
        return fl.common.ndarrays_to_parameters(
            [val.cpu().numpy() for _, val in self.model.state_dict().items()]
        )

    def configure_fit(self, server_round, parameters, client_manager):
        print(f"Configuring fit, sampling 1 client (iteration {server_round})")
        try:
            clients = client_manager.sample(num_clients=1, min_num_clients=1)
            print(f"Sampled {len(clients)} client(s)")
            return [(client, fl.common.FitIns(parameters, {})) for client in clients]
        except Exception as e:
            print(f"Error sampling clients: {e}")
            return []

    def aggregate_fit(self, server_round, results, failures):
        print(f"Aggregating fit results (iteration {server_round})")
        if not results:
            print("No results received")
            return None, {}
        if failures:
            print(f"Failures occurred: {failures}")
            return None, {}

        # Process single client's weights
        client, fit_res = results[0]
        print("Received weights from client")
        try:
            client_weights = fl.common.parameters_to_ndarrays(fit_res.parameters)
        except Exception as e:
            print(f"Error converting client weights: {e}")
            return None, {}

        # Update global model
        try:
            params_dict = zip(self.model.state_dict().keys(), client_weights)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            self.model.load_state_dict(state_dict, strict=True)
        except Exception as e:
            print(f"Error updating model: {e}")
            return None, {}

        # Save model to .h5
        try:
            with h5py.File(self.save_path, 'w') as f:
                for key, param in self.model.state_dict().items():
                    f.create_dataset(key, data=param.cpu().numpy())
            print(f"Saved model to {self.save_path} after client update")
        except Exception as e:
            print(f"Error saving model: {e}")
            return None, {}

        # Return updated parameters
        updated_parameters = fl.common.ndarrays_to_parameters(
            [val.cpu().numpy() for _, val in self.model.state_dict().items()]
        )
        return updated_parameters, {}

    def configure_evaluate(self, server_round, parameters, client_manager):
        print("No evaluation configured")
        return []

    def aggregate_evaluate(self, server_round, results, failures):
        print("No evaluation aggregation")
        return None, {}

    def evaluate(self, server_round, parameters):
        print("No server-side evaluation")
        return None
    def save_model(model, path="model.pt"):
        torch.save(model.state_dict(), path)
        logger.info(f"Saved model to {path}")
class SaveModelFedAvg(fl.server.strategy.FedAvg):
    def __init__(self, model, save_path="model_round_{}.h5", **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.save_path = save_path

    def aggregate_fit(self, server_round, results, failures):
        # Perform standard FedAvg aggregation
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)
        
        if aggregated_parameters is not None:
            # Convert aggregated parameters to model state dict
            try:
                logger.info(f"Saving model after round {server_round}")
                aggregated_ndarrays = fl.common.parameters_to_ndarrays(aggregated_parameters)
                params_dict = zip(self.model.state_dict().keys(), aggregated_ndarrays)
                state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
                self.model.load_state_dict(state_dict, strict=True)

                # Save model to .h5 file
                with h5py.File(self.save_path.format(server_round), 'w') as f:
                    for key, param in self.model.state_dict().items():
                        f.create_dataset(key, data=param.cpu().numpy())
                logger.info(f"Saved model to {self.save_path.format(server_round)}")
            except Exception as e:
                logger.error(f"Error saving model: {e}")

        return aggregated_parameters, aggregated_metrics

if __name__ == "__main__":
    # Initialize model
    model = ViTForAlzheimers(num_labels=4)
    
    # Define custom FedAvg strategy
    strategy = SaveModelFedAvg(
        model=model,
        save_path="model_round_{}.h5",
        min_available_clients=2,
        min_fit_clients=2,
        min_evaluate_clients=2,
    )
    
    # Start server
    logger.info("Starting Flower server...")
    try:
        fl.server.start_server(
            server_address="localhost:8080",
            config=fl.server.ServerConfig(num_rounds=10),
            strategy=strategy
        )
    except KeyboardInterrupt:
        logger.info("Flower server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")