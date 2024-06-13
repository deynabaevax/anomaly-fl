import flwr as fl
from typing import List, Tuple, Dict
import os

os.environ['CURL_CA_BUNDLE'] = ''
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

NUM_OF_CLIENTS = 5
NUM_ROUNDS = 10

def weighted_average(metrics: List[Tuple[int, Dict]]) -> Dict:
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    precisions = [num_examples * m["precision"] for num_examples, m in metrics]
    recalls = [num_examples * m["recall"] for num_examples, m in metrics]
    f1_scores = [num_examples * m["f1_score"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    return {
        "accuracy": sum(accuracies) / sum(examples),
        "precision": sum(precisions) / sum(examples),
        "recall": sum(recalls) / sum(examples),
        "f1_score": sum(f1_scores) / sum(examples),
    }

strategy = fl.server.strategy.FedAvg(
    evaluate_metrics_aggregation_fn=weighted_average,
    fraction_evaluate=1.0,
    fraction_fit=1.0,
    min_fit_clients=NUM_OF_CLIENTS,
    min_evaluate_clients=NUM_OF_CLIENTS,
    min_available_clients=NUM_OF_CLIENTS,
)

fl.server.start_server(
    server_address="localhost:8080",
    config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
    strategy=strategy,
)
