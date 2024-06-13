from typing import List, Tuple
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

# Define your LSTMNet model for anomaly detection
class LSTMNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int) -> None:
        super(LSTMNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        out = torch.sigmoid(out)
        return out

def preprocess_name(name: str) -> int:
    return int(name[1:])

def partition_data(df, num_clients):
    dfs = np.array_split(df, num_clients)
    return dfs

def load_data(csv_path: str, num_clients: int, test_size: float = 0.2, random_state: int = 42, portion: float = 0.1) -> List[Tuple[DataLoader, DataLoader]]:
    df = pd.read_csv(csv_path)

    if df.isnull().sum().sum() > 0:
        print("Missing values found. Handling missing values...")
        df.fillna(df.median(), inplace=True)

    dfs = partition_data(df, num_clients)

    train_loaders = []
    test_loaders = []

    for i in range(num_clients):
        if portion < 1.0:
            df_sampled = dfs[i].sample(frac=portion, random_state=random_state)
        else:
            df_sampled = dfs[i]

        X = df_sampled.drop(columns=['isFraud'])
        y = df_sampled['isFraud']

        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, random_state=random_state)

        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train.to_numpy())
        X_test_tensor = torch.FloatTensor(X_test)
        y_test_tensor = torch.FloatTensor(y_test.to_numpy())

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        train_loader = DataLoader(dataset=train_dataset, batch_size=16, shuffle=True)
        test_loader = DataLoader(dataset=test_dataset, batch_size=16)

        train_loaders.append(train_loader)
        test_loaders.append(test_loader)

    # Return a single tuple containing both lists
    return train_loaders, test_loaders

def create_train_loader(df: pd.DataFrame, test_size: float = 0.2, batch_size: int = 32) -> DataLoader:
    X = df.drop(columns=['isFraud']).values.astype(np.float32)
    y = df['isFraud'].values.astype(np.float32)

    X_train, _, y_train, _ = train_test_split(X, y, test_size=test_size, random_state=42)

    X_train_tensor = torch.tensor(X_train)
    y_train_tensor = torch.tensor(y_train)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    return train_loader

def create_test_loader(df: pd.DataFrame, test_size: float = 0.2, batch_size: int = 32) -> DataLoader:
    X = df.drop(columns=['isFraud']).values.astype(np.float32)
    y = df['isFraud'].values.astype(np.float32)

    _, X_test, _, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    X_test_tensor = torch.tensor(X_test)
    y_test_tensor = torch.tensor(y_test)

    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return test_loader

def disable_progress_bar():
    import os
    os.environ['TQDM_DISABLE'] = '1'