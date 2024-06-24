import argparse
import os
import csv
import glob
import pandas as pd
import numpy as np
import flwr as fl
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import DataLoader
from datasets.utils.logging import disable_progress_bar
from typing import List, Dict, Tuple

from anomaly import load_data

disable_progress_bar()

class AnomalyClient(fl.client.NumPyClient):
    def __init__(self, isolation_forest_model, trainloader: DataLoader, testloader: DataLoader, client_id) -> None:
        super().__init__()
        self.isolation_forest_model = isolation_forest_model
        self.trainloader = trainloader
        self.testloader = testloader
        self.client_id = client_id
        self.approx_round_number = 1

    def log_metric(self, metric_name, value, client_id):
        os.makedirs('metrics', exist_ok=True)
        filename = f"metrics/metrics_client_{client_id}.csv"
        with open(filename, "a", newline='') as file:
            writer = csv.writer(file)
            writer.writerow([metric_name, value, client_id, self.approx_round_number])

    def fit(self, parameters: List[np.ndarray], config: Dict[str, str]) -> Tuple[List[np.ndarray], int, Dict]:
        self.isolation_forest_model.fit(self.trainloader.dataset.tensors[0].numpy())
        training_labels = self.trainloader.dataset.tensors[1].numpy()
        if_train_scores = self.isolation_forest_model.decision_function(self.trainloader.dataset.tensors[0].numpy())
        training_preds = (if_train_scores < 0).astype(int)

        training_loss = 1 - accuracy_score(training_labels, training_preds)
        accuracy = accuracy_score(training_labels, training_preds)
        precision = precision_score(training_labels, training_preds)
        recall = recall_score(training_labels, training_preds)
        f1 = f1_score(training_labels, training_preds)

        self.log_metric("TRAIN_LOSS", training_loss, self.client_id)
        self.log_metric("TRAIN_ACCURACY", accuracy, self.client_id)
        self.log_metric("TRAIN_PRECISION", precision, self.client_id)
        self.log_metric("TRAIN_RECALL", recall, self.client_id)
        self.log_metric("TRAIN_F1_SCORE", f1, self.client_id)

        return [], len(self.trainloader.dataset), {
            "training_loss": training_loss,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }

    def evaluate(self, parameters: List[np.ndarray], config: Dict[str, str]) -> Tuple[float, int, Dict]:
        test_labels = self.testloader.dataset.tensors[1].numpy()
        if_test_scores = self.isolation_forest_model.decision_function(self.testloader.dataset.tensors[0].numpy())
        test_preds = (if_test_scores < 0).astype(int)

        testing_loss = 1 - accuracy_score(test_labels, test_preds)
        accuracy = accuracy_score(test_labels, test_preds)
        precision = precision_score(test_labels, test_preds)
        recall = recall_score(test_labels, test_preds)
        f1 = f1_score(test_labels, test_preds)

        metrics = {
            "testing_loss": testing_loss,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }

        for metric_name, value in metrics.items():
            self.log_metric(metric_name.upper(), value, self.client_id)

        self.approx_round_number += 1

        return testing_loss, len(self.testloader.dataset), metrics


def aggregate_metrics():
    all_files = glob.glob('metrics/*.csv')
    df_list = []
    for filename in all_files:
        df = pd.read_csv(filename, names=['Metric', 'Value', 'ClientID', 'Round'])
        df_list.append(df)
    return pd.concat(df_list, ignore_index=True)

def plot_metric_per_round(metrics_df, metric_name, title, save_path):
    filtered_df = metrics_df[metrics_df['Metric'] == metric_name]
    if filtered_df.empty:
        print(f"No data found for metric '{metric_name}'")
        return
    pivot_df = filtered_df.pivot_table(index='Round', columns='ClientID', values='Value', aggfunc='mean')

    plt.figure(figsize=(10, 6))
    for column in pivot_df.columns:
        plt.plot(pivot_df.index, pivot_df[column], marker='o', label=f'Client {column}')

    plt.title(title)
    plt.xlabel('Federated Round')
    plt.ylabel(metric_name.replace("_", " ").capitalize())
    plt.legend(title='Client ID')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_convergence(metrics_df, metric_name, save_path):
    filtered_df = metrics_df[metrics_df['Metric'] == metric_name]
    if filtered_df.empty:
        print(f"No data found for metric '{metric_name}'")
        return
    
    plt.figure(figsize=(10, 6))
    for client_id in filtered_df['ClientID'].unique():
        client_data = filtered_df[filtered_df['ClientID'] == client_id]
        plt.plot(client_data['Round'], client_data['Value'], marker='o', label=f'Client {client_id}')
    
    plt.title(f'{metric_name.replace("_", " ").capitalize()} Convergence')
    plt.xlabel('Federated Round')
    plt.ylabel(metric_name.replace("_", " ").capitalize())
    plt.legend(title='Client ID')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    
def main():
    parser = argparse.ArgumentParser(description="FL client for anomaly detection")
    parser.add_argument("--server_address", type=str, default="localhost:8080", help="Server address")
    parser.add_argument("--client_id", type=int, default=0, help="Client ID")
    parser.add_argument("--node-id", type=int, required=True, choices=range(0, 5), help="Node ID")
    args = parser.parse_args()

    num_clients = 5
    train_loaders, test_loaders = load_data("C:/Users/deyna/Desktop/anomaly-fl-main/data/feat.csv", num_clients=num_clients)

    isolation_forest_model = IsolationForest(contamination=0.1)

    clients = []
    for client_id in range(num_clients):
        client = AnomalyClient(isolation_forest_model, train_loaders[client_id], test_loaders[client_id], client_id).to_client()
        clients.append(client)

    client_ip = "localhost"
    client_port = "8080"
    server_address = f"{client_ip}:{client_port}"
    print(f"Connecting to {server_address}")
    fl.client.start_client(server_address=server_address, client=clients[args.client_id])

    metrics_df = aggregate_metrics()
    metrics_df.to_csv("metrics/metrics_aggregated.csv", index=False)
    print("Metrics DataFrame saved to metrics/metrics_aggregated.csv")
    print(metrics_df.head())

    plot_metric_per_round(metrics_df, 'TRAIN_ACCURACY', 'Average Training Accuracy per Round', 'metrics/training_accuracy_per_round.png')
    plot_convergence(metrics_df, 'ACCURACY', 'metrics/accuracy_convergence.png')

if __name__ == "__main__":
    main()
    

'''
# old code
class AnomalyClient(fl.client.NumPyClient):
    def __init__(self, isolation_forest_model, trainloader: DataLoader, testloader: DataLoader, client_id) -> None:
        super().__init__()
        self.isolation_forest_model = isolation_forest_model
        self.trainloader = trainloader
        self.testloader = testloader
        self.client_id = client_id
        self.approx_round_number = 1  # Initialize round number for this client instance

    def log_metric(self, metric_name, value):
        os.makedirs('metrics', exist_ok=True)
        # Use client_id for unique filename
        filename = f"metrics/metrics_client_{self.client_id}.csv" 
        with open(filename, "a", newline='') as file:
            writer = csv.writer(file)
            writer.writerow([metric_name, value, self.client_id, self.approx_round_number]) 

    def fit(self, parameters: List[np.ndarray], config: Dict[str, str]) -> Tuple[List[np.ndarray], int, Dict]:
        self.isolation_forest_model.fit(self.trainloader.dataset.tensors[0].numpy())
        training_labels = self.trainloader.dataset.tensors[1].numpy()
        if_train_scores = self.isolation_forest_model.decision_function(self.trainloader.dataset.tensors[0].numpy())
        training_preds = (if_train_scores < 0).astype(int)

        training_loss = 1 - accuracy_score(training_labels, training_preds)
        accuracy = accuracy_score(training_labels, training_preds)
        precision = precision_score(training_labels, training_preds)
        recall = recall_score(training_labels, training_preds)
        f1 = f1_score(training_labels, training_preds)

        self.log_metric("TRAIN_LOSS", training_loss)
        self.log_metric("TRAIN_ACCURACY", accuracy)
        self.log_metric("TRAIN_PRECISION", precision)
        self.log_metric("TRAIN_RECALL", recall)
        self.log_metric("TRAIN_F1_SCORE", f1)

        # # Increment round number after each fit
        # self.approx_round_number += 1

        return [], len(self.trainloader.dataset), {
            "training_loss": training_loss,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }

    def evaluate(self, parameters: List[np.ndarray], config: Dict[str, str]) -> Tuple[float, int, Dict]:
        test_labels = self.testloader.dataset.tensors[1].numpy()
        if_test_scores = self.isolation_forest_model.decision_function(self.testloader.dataset.tensors[0].numpy())
        test_preds = (if_test_scores < 0).astype(int)

        testing_loss = 1 - accuracy_score(test_labels, test_preds)
        accuracy = accuracy_score(test_labels, test_preds)
        precision = precision_score(test_labels, test_preds)
        recall = recall_score(test_labels, test_preds)
        f1 = f1_score(test_labels, test_preds)

        self.log_metric("TESTING_LOSS", testing_loss)
        self.log_metric("TEST_ACCURACY", accuracy)
        self.log_metric("TEST_PRECISION", precision)
        self.log_metric("TEST_RECALL", recall)
        self.log_metric("TEST_F1_SCORE", f1)

        # Increment round number after each evaluate
        self.approx_round_number += 1

        return testing_loss, len(self.testloader.dataset), {
            "testing_loss": testing_loss,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }

def aggregate_metrics():
    all_files = glob.glob('metrics/*.csv')
    df_list = []
    for filename in all_files:
        df = pd.read_csv(filename, names=['Metric', 'Value', 'ClientID', 'Round'])
        df_list.append(df)
    return pd.concat(df_list, ignore_index=True)
    
def plot_metric_per_round(metrics_df, metric_name, title, save_path):
    filtered_df = metrics_df[metrics_df['Metric'] == metric_name]
    if filtered_df.empty:
        print(f"No data found for metric '{metric_name}'")
        return
    pivot_df = filtered_df.pivot_table(index='Round', columns='ClientID', values='Value', aggfunc='mean')

    plt.figure(figsize=(10, 6))
    for column in pivot_df.columns:
        plt.plot(pivot_df.index, pivot_df[column], marker='o', label=f'Client {column}')

    plt.title(title)
    plt.xlabel('Federated Round')
    plt.ylabel(metric_name.replace("_", " ").capitalize())
    plt.legend(title='Client ID')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="FL client for anomaly detection")
    parser.add_argument("--server_address", type=str, default="localhost:8080", help="Server address")
    parser.add_argument("--client_id", type=int, default=0, help="Client ID")
    parser.add_argument("--node-id", type=int, required=True, choices=range(0, 5), help="Node ID")
    args = parser.parse_args()

    num_clients = 5
    
    # Replace the data.csv with your own file
    train_loaders, test_loaders = load_data("C:/Users/deyna/Desktop/anomaly-fl-main/data/feat.csv", num_clients=num_clients)

    isolation_forest_model = IsolationForest(contamination=0.1)

    clients = []
    for client_id in range(num_clients):
        client = AnomalyClient(isolation_forest_model, train_loaders[client_id], test_loaders[client_id], client_id).to_client()
        clients.append(client)

    client_ip = "localhost"
    client_port = "8080"
    server_address = f"{client_ip}:{client_port}"
    print(f"Connecting to {server_address}")
    fl.client.start_client(server_address=server_address, client=clients[args.client_id])

    metrics_df = aggregate_metrics()
    metrics_df.to_csv("metrics/metrics_aggregated.csv", index=False)
    print("Metrics DataFrame saved to metrics/metrics_aggregated.csv")
    print(metrics_df.head())

    plot_metric_per_round(metrics_df, 'TRAIN_ACCURACY', 'Average Training Accuracy per Round', 'metrics/training_accuracy_per_round.png')


if __name__ == "__main__":
    main() '''  


# with both models

'''from anomaly import load_data

disable_progress_bar()

class AnomalyClient(fl.client.NumPyClient):
    def __init__(self, isolation_forest_model, trainloader: DataLoader, testloader: DataLoader, client_id) -> None:
        super().__init__()
        self.isolation_forest_model = isolation_forest_model
        self.trainloader = trainloader
        self.testloader = testloader
        self.client_id = client_id
        self.approx_round_number = 1

    def get_parameters(self, config=None) -> List[np.ndarray]:
        # Isolation Forest does not have parameters to get
        return []

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        # Isolation Forest does not have parameters to set
        pass

    def log_metric(self, metric_name, value):
        os.makedirs('metrics', exist_ok=True)
        filename = f"metrics/metrics_client_{self.client_id}.csv"
        with open(filename, "a", newline='') as file:
            writer = csv.writer(file)
            writer.writerow([metric_name, value, self.client_id, self.approx_round_number])

    def fit(self, parameters: List[np.ndarray], config: Dict[str, str]) -> Tuple[List[np.ndarray], int, Dict]:
        # Train the Isolation Forest model
        self.isolation_forest_model.fit(self.trainloader.dataset.tensors[0].numpy())
        
        # Evaluate the model on training data
        training_labels = self.trainloader.dataset.tensors[1].numpy()
        if_train_scores = self.isolation_forest_model.decision_function(self.trainloader.dataset.tensors[0].numpy())
        training_preds = (if_train_scores < 0).astype(int)

        training_loss = 1 - accuracy_score(training_labels, training_preds)
        accuracy = accuracy_score(training_labels, training_preds)
        precision = precision_score(training_labels, training_preds)
        recall = recall_score(training_labels, training_preds)
        f1 = f1_score(training_labels, training_preds)
        
        self.log_metric("TRAIN_LOSS", training_loss)
        self.log_metric("TRAIN_ACCURACY", accuracy)
        self.log_metric("TRAIN_PRECISION", precision)
        self.log_metric("TRAIN_RECALL", recall)
        self.log_metric("TRAIN_F1_SCORE", f1)

        return [], len(self.trainloader.dataset), {
            "training_loss": training_loss,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }

    def evaluate(self, parameters: List[np.ndarray], config: Dict[str, str]) -> Tuple[float, int, Dict]:
        # Evaluate the model on test data
        test_labels = self.testloader.dataset.tensors[1].numpy()
        if_test_scores = self.isolation_forest_model.decision_function(self.testloader.dataset.tensors[0].numpy())
        test_preds = (if_test_scores < 0).astype(int)

        testing_loss = 1 - accuracy_score(test_labels, test_preds)
        accuracy = accuracy_score(test_labels, test_preds)
        precision = precision_score(test_labels, test_preds)
        recall = recall_score(test_labels, test_preds)
        f1 = f1_score(test_labels, test_preds)
        
        self.approx_round_number += 1
        
        metrics = {
            "testing_loss": testing_loss,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }
        
        self.log_metrics(metrics)
        
        return testing_loss, len(self.testloader.dataset), metrics
        
    def log_metrics(self, metrics):
        os.makedirs('metrics', exist_ok=True)
        filename = f"metrics/metrics_client_{self.client_id}.csv"
        with open(filename, "a", newline='') as file:
            writer = csv.writer(file)
            for metric_name, value in metrics.items():
                writer.writerow([metric_name, value, self.client_id, self.approx_round_number])

def main() -> None:
    parser = argparse.ArgumentParser(description="FL client for anomaly detection")
    parser.add_argument("--server_address", type=str, default="localhost:8080", help="Server address")
    parser.add_argument("--client_id", type=int, default=0, help="Client ID")
    parser.add_argument("--node-id", type=int, required=True, choices=range(0, 5), help="Node ID")
    args = parser.parse_args()
    
    # Load data for this client
    num_clients = 5 
    train_loaders, test_loaders = load_data("C:/Users/deyna/Desktop/anomaly-fl-main/data/feat.csv", num_clients=num_clients)

    # Initialize the Isolation Forest model and client
    isolation_forest_model = IsolationForest(contamination=0.1)
    client = AnomalyClient(isolation_forest_model, train_loaders[args.client_id], test_loaders[args.client_id], args.client_id)
    
    if isinstance(client, fl.client.NumPyClient):
        client = client.to_client()

    #  Start the FL client
    client_ip = "localhost"
    client_port = "8080"
    
    server_address = f"{client_ip}:{client_port}"
    print(f"Connecting to {server_address}")
    fl.client.start_client(server_address=server_address, client=client)

if __name__ == "__main__":
    main()'''
