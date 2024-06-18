import argparse
import os
import csv
import glob
import pandas as pd
import numpy as np
import flwr as fl
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import DataLoader
from datasets.utils.logging import disable_progress_bar
from typing import List, Dict, Tuple

from anomaly import load_data_in_chunks

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

def plot_feature_distributions(train_loaders):
    dfs = [pd.DataFrame(client.dataset.tensors[0].numpy(), columns=[
        'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest',
        'changebalanceOrig', 'changebalanceDest', 'hour'
    ]) for client in train_loaders]
    df_all_clients = pd.concat(dfs, ignore_index=True)
    
    numerical_features = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest', 
                          'changebalanceOrig', 'changebalanceDest', 'hour']
    
    plt.figure(figsize=(20, 16))
    for i, feature in enumerate(numerical_features, start=1):
        plt.subplot(4, 2, i)
        for client_id, client_data in enumerate(dfs, start=1):
            sns.histplot(client_data[feature], bins=30, kde=True, label=f'Client {client_id}', alpha=0.5)
        plt.title(f'Distribution of {feature}')
        plt.xlabel(feature)
        plt.ylabel('Density')
        plt.legend()
    plt.tight_layout()
    plt.savefig('metrics/feature_distributions.png', dpi=300)
    plt.close()
    
def plot_label_distribution(train_loaders):
    dfs = [pd.DataFrame(client.dataset.tensors[1].numpy(), columns=['isFraud']) 
           for client in train_loaders]
    
    plt.figure(figsize=(10, 8))
    fraud_proportions = []
    for client_id, client_data in enumerate(dfs, start=1):
        fraud_counts = client_data['isFraud'].value_counts(normalize=True)
        fraud_proportions.append((client_id, fraud_counts.get(1, 0), fraud_counts.get(0, 0)))
    
    fraud_proportions_df = pd.DataFrame(fraud_proportions, columns=['ClientID', 'Fraudulent', 'Non-Fraudulent'])
    fraud_proportions_df.set_index('ClientID').plot(kind='bar', stacked=True, color=['red', 'blue'], alpha=0.6)
    
    plt.title('Distribution of Fraudulent vs Non-Fraudulent Transactions')
    plt.xlabel('Client')
    plt.ylabel('Proportion')
    plt.legend(['Fraudulent', 'Non-Fraudulent'])
    plt.tight_layout()
    plt.savefig('metrics/label_distribution.png', dpi=300)
    plt.close()

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

    # Define the fraud ratios for each client
    # This is helping introduce even more heterogeneity in the data
    fraud_ratios = [0.2, 0.3, 0.1, 0.4, 0.5]

    train_loaders, test_loaders = load_data_in_chunks(
        "data.csv",
        num_clients=num_clients,
        fraud_ratios=fraud_ratios
    )

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

    plot_feature_distributions(train_loaders)
    plot_label_distribution(train_loaders)    

if __name__ == "__main__":
    main()


'''
# OCSVM
import argparse
import csv
import glob
import os
from typing import Dict, List, Tuple

import flwr as fl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datasets.utils.logging import disable_progress_bar
# from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)
# from sklearn.neighbors import LocalOutlierFactor
from torch.utils.data import DataLoader

from anomaly import load_data

disable_progress_bar()

class AnomalyClient(fl.client.NumPyClient):
    def __init__(self, ocsvm_model, trainloader: DataLoader, testloader: DataLoader, client_id) -> None:
        super().__init__()
        self.ocsvm_model = ocsvm_model
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
        self.ocsvm_model.fit(self.trainloader.dataset.tensors[0].numpy())

        # Predict anomalies with OCSVM
        training_scores = self.ocsvm_model.score_samples(self.trainloader.dataset.tensors[0].numpy())
        training_preds = (training_scores < 0).astype(int)

        training_loss = 1 - accuracy_score(self.trainloader.dataset.tensors[1].numpy(), training_preds)
        accuracy = accuracy_score(self.trainloader.dataset.tensors[1].numpy(), training_preds)
        precision = precision_score(self.trainloader.dataset.tensors[1].numpy(), training_preds)
        recall = recall_score(self.trainloader.dataset.tensors[1].numpy(), training_preds)
        f1 = f1_score(self.trainloader.dataset.tensors[1].numpy(), training_preds)

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
        # Predict anomalies with OCSVM
        test_scores = self.ocsvm_model.score_samples(self.testloader.dataset.tensors[0].numpy())
        test_preds = (test_scores < 0).astype(int)

        # Calculate metrics
        testing_loss = 1 - accuracy_score(self.testloader.dataset.tensors[1].numpy(), test_preds)
        accuracy = accuracy_score(self.testloader.dataset.tensors[1].numpy(), test_preds)
        precision = precision_score(self.testloader.dataset.tensors[1].numpy(), test_preds)
        recall = recall_score(self.testloader.dataset.tensors[1].numpy(), test_preds)
        f1 = f1_score(self.testloader.dataset.tensors[1].numpy(), test_preds)

        # Log metrics
        self.log_metric("TEST_LOSS", testing_loss, self.client_id)
        self.log_metric("TEST_ACCURACY", accuracy, self.client_id)
        self.log_metric("TEST_PRECISION", precision, self.client_id)
        self.log_metric("TEST_RECALL", recall, self.client_id)
        self.log_metric("TEST_F1_SCORE", f1, self.client_id)

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

    # isolation_forest_model = IsolationForest(contamination=0.1)
    # lof_model = LocalOutlierFactor(contamination=0.1)
    ocsvm_model = OneClassSVM(kernel='rbf', nu=0.1, gamma=0.1)

    clients = []
    for client_id in range(num_clients):
        client = AnomalyClient(ocsvm_model, train_loaders[client_id], test_loaders[client_id], client_id).to_client()
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
    main()'''
    
    
'''
# with both models
class AnomalyClient(fl.client.NumPyClient):
    def __init__(self, isolation_forest_model, lof_model, trainloader: DataLoader, testloader: DataLoader, client_id) -> None:
        super().__init__()
        self.isolation_forest_model = isolation_forest_model
        self.lof_model = lof_model
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
        X_train = self.trainloader.dataset.tensors[0].numpy()
        y_train = self.trainloader.dataset.tensors[1].numpy()

        # Step 1: Initial pruning with Isolation Forest
        self.isolation_forest_model.fit(X_train)
        if_scores = self.isolation_forest_model.decision_function(X_train)
        if_preds = (if_scores < 0).astype(int)

        # Step 2: Generate outlier candidate set
        candidate_indices = np.where(if_preds == 1)[0]
        X_candidates = X_train[candidate_indices]

        # Step 3: Refinement with LOF
        if len(X_candidates) > 0:
            self.lof_model.fit(X_candidates)
            lof_scores = -self.lof_model.negative_outlier_factor_
            lof_preds = (lof_scores < -1.5).astype(int)
        else:
            lof_preds = np.array([])

        # Combining the results
        refined_preds = np.zeros_like(if_preds)
        if len(lof_preds) > 0:
            refined_preds[candidate_indices] = lof_preds

        # Calculate training metrics
        training_loss = 1 - accuracy_score(y_train, refined_preds)
        accuracy = accuracy_score(y_train, refined_preds)
        precision = precision_score(y_train, refined_preds, zero_division=0)
        recall = recall_score(y_train, refined_preds, zero_division=0)
        f1 = f1_score(y_train, refined_preds, zero_division=0)

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
        X_test = self.testloader.dataset.tensors[0].numpy()
        
        # Evaluate with Isolation Forest
        if_test_scores = self.isolation_forest_model.decision_function(X_test)
        test_preds_if = (if_test_scores < 0).astype(int)
        print(f"Isolation Forest predictions shape: {test_preds_if.shape}")
        
        # Evaluate with LOF
        test_scores_lof = -self.lof_model.negative_outlier_factor_
        test_preds_lof = (test_scores_lof < -1.5).astype(int)
        print(f"LOF predictions shape: {test_preds_lof.shape}")
        
        # Combine predictions for evaluation
        test_preds_combined = np.logical_or(test_preds_if, test_preds_lof).astype(int)
        
        test_labels = self.testloader.dataset.tensors[1].numpy()

        # Calculate metrics and log
        testing_loss = 1 - accuracy_score(test_labels, test_preds_combined)
        accuracy = accuracy_score(test_labels, test_preds_combined)
        precision = precision_score(test_labels, test_preds_combined)
        recall = recall_score(test_labels, test_preds_combined)
        f1 = f1_score(test_labels, test_preds_combined)

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
    
    plt.figure(figsize=(10, 6))
    for client_id in filtered_df['ClientID'].unique():
        client_data = filtered_df[filtered_df['ClientID'] == client_id]
        plt.plot(client_data['Round'], client_data['Value'], marker='o', label=f'Client {client_id}')
    
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
    lof_model = LocalOutlierFactor(contamination=0.1) 

    clients = []
    for client_id in range(num_clients):
        client = AnomalyClient(isolation_forest_model, lof_model, train_loaders[client_id], test_loaders[client_id], client_id).to_client()
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