import argparse
import csv
import os
from typing import Dict, List, Tuple
import flwr as fl
import numpy as np
import pandas as pd
import torch
from datasets.utils.logging import disable_progress_bar
from torch.utils.data import DataLoader
import torch.nn as nn
from sklearn.metrics import precision_score, recall_score, f1_score

# Import AnomalyClient from anomaly module
from anomaly import LSTMNet, load_data

disable_progress_bar()

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class AnomalyClient(fl.client.NumPyClient):
    def __init__(self, model: LSTMNet, trainloader: DataLoader, testloader: DataLoader, client_id) -> None:
        super().__init__()
        self.model = model
        self.trainloader = trainloader
        self.testloader = testloader
        self.client_id = client_id
        self.approx_round_number = 1

    def get_parameters(self, config=None) -> List[np.ndarray]:
        self.model.train()
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        self.model.train()
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def log_metric(self, metric_name, value):
        os.makedirs('metrics', exist_ok=True)
        filename = f"metrics/metrics_client_{self.client_id}.csv"
        with open(filename, "a", newline='') as file:
            writer = csv.writer(file)
            writer.writerow([metric_name, value, self.client_id, self.approx_round_number])

    def fit(self, parameters: List[np.ndarray], config: Dict[str, str]) -> Tuple[List[np.ndarray], int, Dict]:
        self.set_parameters(parameters)
        training_loss, accuracy = train(self.model, self.trainloader, epochs=5, device=DEVICE)

        self.log_metric("TRAIN_LOSS", training_loss)
        self.log_metric("TRAIN_ACCURACY", accuracy)

        return self.get_parameters(), len(self.trainloader.dataset), {
            "training_loss": training_loss,
            "accuracy": accuracy,
        }

'''    def evaluate(self, parameters: List[np.ndarray], config: Dict[str, str]) -> Tuple[float, int, Dict]:
        self.set_parameters(parameters)
        testing_loss, accuracy = test(self.model, self.testloader, device=DEVICE)
        self.approx_round_number += 1
        return testing_loss, len(self.testloader.dataset), {"testing_loss": testing_loss, "accuracy": accuracy}'''
        
def evaluate(self, parameters: List[np.ndarray], config: Dict[str, str]) -> Tuple[float, int, Dict]:
    self.set_parameters(parameters)
    testing_loss, accuracy, precision, recall, f1 = test(self.model, self.testloader, device=DEVICE)
    self.approx_round_number += 1
    
    metrics = {
        "testing_loss": testing_loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }
    
    log_metrics(self.client_id, self.approx_round_number, metrics)
    
    return testing_loss, len(self.testloader.dataset), metrics
        
def log_metrics(client_id, round_number, metrics):
    os.makedirs('metrics', exist_ok=True)
    filename = f"metrics/metrics_client_{client_id}.csv"
    with open(filename, "a", newline='') as file:
        writer = csv.writer(file)
        for metric_name, value in metrics.items():
            writer.writerow([metric_name, value, client_id, round_number])

def train(model: LSTMNet, trainloader: DataLoader, epochs: int, device: torch.device) -> Tuple[float, float]:
    model.train()
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    total_loss, correct = 0.0, 0
    for epoch in range(epochs):
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            predicted = (outputs > 0.5).float()
            correct += (predicted == labels).sum().item()

    accuracy = correct / len(trainloader.dataset)
    return total_loss / len(trainloader), accuracy

def test(model: LSTMNet, testloader: DataLoader, device: torch.device) -> Tuple[float, float, float, float, float]:
    model.eval()
    criterion = nn.BCELoss()

    total_loss, correct = 0.0, 0
    all_labels = []
    all_predictions = []
    
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            predicted = (outputs > 0.5).float()
            correct += (predicted == labels).sum().item()
            
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    accuracy = correct / len(testloader.dataset)
    precision = precision_score(all_labels, all_predictions)
    recall = recall_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions)
    
    return total_loss / len(testloader), accuracy, precision, recall, f1

    
# def main() -> None:
#     parser = argparse.ArgumentParser(description="FL client for anomaly detection")
#     parser.add_argument("--server_address", type=str, default="localhost:8080", help="Server address")
#     parser.add_argument("--client_id", type=int, default=0, help="Client ID")
#     parser.add_argument("--node-id", type=int, required=True, choices=range(0, 5), help="Node ID")
#     args = parser.parse_args()
    
#     # Load data for this client
#     num_clients = 5 
#     train_loaders, test_loaders = load_data("C:/Users/deyna/Desktop/anomaly-fl-main/data/feat.csv", num_clients=num_clients)

#     # Initialize the model and client
#     input_dim = len(train_loaders[0].dataset[0][0])
#     hidden_dim = 32
#     num_layers = 2
#     model = LSTMNet(input_dim, hidden_dim, num_layers).to(DEVICE)
#     client = AnomalyClient(model, train_loaders[args.client_id], test_loaders[args.client_id], args.client_id)

#     # Start the FL client
#     client_ip, client_port = args.server_address.split(":")
#     client_port = int(client_port)
#     client.log_metric("START", 1)
#     fl.client.start_numpy_client(server_address=f"grpc://{client_ip}:{client_port}", client=client)

# def main():
#     parser = argparse.ArgumentParser(description="FL client for anomaly detection")
#     parser.add_argument("--server_address", type=str, default="localhost:8080", help="Server address")
#     parser.add_argument("--client_id", type=int, default=0, help="Client ID")
#     parser.add_argument("--node-id", type=int, required=True, choices=range(0, 5), help="Node ID")
#     args = parser.parse_args()
    
#     # Load data for this client
#     num_clients = 5 
#     train_loaders, test_loaders = load_data("C:/Users/deyna/Desktop/anomaly-fl-main/data/feat.csv", num_clients=num_clients)

#     # Initialize the model and client
#     input_dim = len(train_loaders[0].dataset[0][0])
#     hidden_dim = 32
#     num_layers = 2
#     model = LSTMNet(input_dim, hidden_dim, num_layers).to(DEVICE)
#     client = AnomalyClient(model, train_loaders[args.client_id], test_loaders[args.client_id], args.client_id)

#     # Start the FL client
#     client_ip = "localhost"  # Replace with actual client IP address if necessary
#     client_port = "8080"      # Replace with actual client port if necessary
    
#     server_address = f"grpc://{args.server_address}"
#     print(f"Connecting to {server_address}")
#     client.log_metric("START", 1)

#     # Initialize client and start connection
#     fl.client.start_numpy_client(server_address=server_address, client=client)

def main() -> None:
    parser = argparse.ArgumentParser(description="FL client for anomaly detection")
    parser.add_argument("--server_address", type=str, default="localhost:8080", help="Server address")
    parser.add_argument("--client_id", type=int, default=0, help="Client ID")
    # Add --node-id argument 
    parser.add_argument("--node-id", type=int, required=True, choices=range(0, 5), help="Node ID")
    args = parser.parse_args()
    
    # Load data for this client
    num_clients = 5 
    train_loaders, test_loaders = load_data("C:/Users/deyna/Desktop/anomaly-fl-main/data/feat.csv", num_clients=num_clients)

    # Initialize the model and client
    input_dim = len(train_loaders[0].dataset[0][0])
    hidden_dim = 32
    num_layers = 2
    model = LSTMNet(input_dim, hidden_dim, num_layers).to(DEVICE)
    client = AnomalyClient(model, train_loaders[args.client_id], test_loaders[args.client_id], args.client_id)
    
    if isinstance(client, fl.client.NumPyClient):
        client = client.to_client()

    #  Start the FL client
    client_ip = "localhost"
    client_port = "8080"
    
    server_address = f"{client_ip}:{client_port}"
    print(f"Connecting to {server_address}")
    # client.log_metric("START", 1)
    # Initialise client and start connection
    fl.client.start_client(server_address = server_address, client=client)

if __name__ == "__main__":
    main()