import flwr as fl
from flwr.server.strategy import FedAvg
from flwr.common import Metrics
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
import argparse
from typing import List, Tuple, Optional, Dict
import numpy as np
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy
from flwr.common import FitRes, Parameters

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregate metrics from multiple clients using weighted average"""
    
    # Multiply accuracy/auc of each client by number of samples
    accuracies = [num_samples * m["accuracy"] for num_samples, m in metrics]
    aucs = [num_samples * m["auc"] for num_samples, m in metrics if "auc" in m]
    
    total_samples = sum([num_samples for num_samples, _ in metrics])
    
    # Aggregate and return custom metric (weighted average)
    return {
        "accuracy": sum(accuracies) / total_samples,
        "auc": sum(aucs) / total_samples if aucs else 0.0,
    }


def get_initial_parameters():
    """Initialize model and return its parameters"""
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(nn.Dropout(p=0.3), nn.Linear(num_ftrs, 1))
    
    # Return parameters as list of NumPy arrays
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


class SaveModelStrategy(FedAvg):
    """Custom strategy that saves the global model after each round"""

    def __init__(self, save_path: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_path = save_path
        self.best_auc = 0.0
        self.aggregated_parameters: Optional[Parameters] = None # Attribute to store params

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[Parameters], Dict[str, fl.common.Scalar]]:
        """Aggregate fit results and store the new global model parameters."""
        
        # Call the parent aggregate_fit to perform the aggregation
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )
        
        # Store the aggregated parameters
        if aggregated_parameters is not None:
            self.aggregated_parameters = aggregated_parameters
            
        return aggregated_parameters, aggregated_metrics

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, fl.common.EvaluateRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[float], Dict[str, fl.common.Scalar]]:
        """Aggregate evaluation results and save best model"""
        
        if not results:
            return None, {}
        
        # Call parent aggregate_evaluate to get aggregated metrics
        loss_aggregated, metrics_aggregated = super().aggregate_evaluate(
            server_round, results, failures
        )
        
        # Save model if AUC improved
        if metrics_aggregated and "auc" in metrics_aggregated:
            current_auc = metrics_aggregated["auc"]
            if current_auc > self.best_auc:
                self.best_auc = current_auc
                print(f"\n🎉 New best AUC: {self.best_auc:.4f} (Round {server_round})")
                
                # Check if we have parameters to save
                if self.aggregated_parameters is not None:
                    print("✅ Saving model...")
                    
                    # Convert `Parameters` to a list of NumPy arrays
                    ndarrays = parameters_to_ndarrays(self.aggregated_parameters)
                    
                    # Create a new model instance and load the state
                    model = resnet18()
                    num_ftrs = model.fc.in_features
                    model.fc = nn.Sequential(nn.Dropout(p=0.3), nn.Linear(num_ftrs, 1))
                    
                    params_dict = zip(model.state_dict().keys(), ndarrays)
                    state_dict = {k: torch.tensor(v) for k, v in params_dict}
                    model.load_state_dict(state_dict, strict=True)
                    
                    # Save the state_dict
                    torch.save(model.state_dict(), self.save_path)
                    print(f"✅ Model saved to {self.save_path}")
        
        return loss_aggregated, metrics_aggregated


def main():
    parser = argparse.ArgumentParser(description='Flower Server for MURA')
    parser.add_argument('--num_rounds', type=int, default=10, 
                        help='Number of federated learning rounds')
    parser.add_argument('--num_clients', type=int, default=7,
                        help='Total number of clients')
    parser.add_argument('--min_clients', type=int, default=2,
                        help='Minimum number of clients required per round')
    parser.add_argument('--server_address', type=str, default='0.0.0.0:8080',
                        help='Server address')
    parser.add_argument('--partition_strategy', type=str, required=True,
                        choices=['iid', 'pathological_non_iid', 'label_skew'],
                        help='Partitioning strategy')
    parser.add_argument('--save_path', type=str, default=None,
                        help='Path to save the best model')
    
    args = parser.parse_args()
    
    # Set default save path based on partition strategy
    if args.save_path is None:
        args.save_path = f"best_federated_model_{args.partition_strategy}.pth"
    
    print(f"\n{'='*60}")
    print(f"Starting Flower Server")
    print(f"{'='*60}")
    print(f"Partition Strategy: {args.partition_strategy}")
    print(f"Number of Rounds: {args.num_rounds}")
    print(f"Total Clients: {args.num_clients}")
    print(f"Min Clients per Round: {args.min_clients}")
    print(f"Server Address: {args.server_address}")
    print(f"Model Save Path: {args.save_path}")
    print(f"{'='*60}\n")
    
    # Initialize model parameters
    initial_parameters = get_initial_parameters()
    
    # Define strategy with custom model saving
    strategy = SaveModelStrategy(
        save_path=args.save_path,
        fraction_fit=1.0,  # Use all available clients for training
        fraction_evaluate=1.0,  # Use all available clients for evaluation
        min_fit_clients=args.min_clients,
        min_evaluate_clients=args.min_clients,
        min_available_clients=args.min_clients,
        initial_parameters=fl.common.ndarrays_to_parameters(initial_parameters),
        evaluate_metrics_aggregation_fn=weighted_average,
    )
    
    # Start server
    fl.server.start_server(
        server_address=args.server_address,
        config=fl.server.ServerConfig(num_rounds=args.num_rounds),
        strategy=strategy,
    )
    
    print(f"\n{'='*60}")
    print("Federated Learning Complete!")
    print(f"Best AUC: {strategy.best_auc:.4f}")
    print(f"Best model saved to: {args.save_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()