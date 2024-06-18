import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

def preprocess_name(name: str) -> int:
    return int(name[1:])

def partition_data(df, num_clients):
    dfs = np.array_split(df, num_clients)
    return dfs

def load_data_in_chunks(csv_path: str, num_clients: int, fraud_ratios: list, test_size: float = 0.2, random_state: int = 42):
    df = pd.read_csv(csv_path)

    if df.isnull().sum().sum() > 0:
        print("Missing values found. Handling missing values with mean imputation...")
        df.fillna(df.mean(), inplace=True)

    # Separate fraudulent and non-fraudulent transactions
    fraud_df = df[df['isFraud'] == 1]
    non_fraud_df = df[df['isFraud'] == 0]

    # Split the non-fraudulent data evenly among clients
    non_fraud_splits = np.array_split(non_fraud_df, num_clients)

    train_loaders = []
    test_loaders = []

    for i in range(num_clients):
        # Sample the fraudulent data based on the specified ratio
        fraud_sample = fraud_df.sample(frac=fraud_ratios[i], random_state=random_state)
        
        # Combine the non-fraudulent and fraudulent samples
        client_df = pd.concat([non_fraud_splits[i], fraud_sample])

        # Shuffle the client data
        client_df = client_df.sample(frac=1, random_state=random_state).reset_index(drop=True)

        # Ensure the DataFrame has the correct shape
        assert client_df.shape[1] == df.shape[1], f"Expected {df.shape[1]} columns, got {client_df.shape[1]}"

        # X = client_df[['amount', 'hour']].values
        X = client_df[['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest', 
                        'changebalanceOrig', 'changebalanceDest', 'hour']].values
        y = client_df['isFraud'].values

        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, random_state=random_state)

        # Convert data to PyTorch tensors and create DataLoader objects
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train)
        X_test_tensor = torch.FloatTensor(X_test)
        y_test_tensor = torch.FloatTensor(y_test)

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        train_loader = DataLoader(dataset=train_dataset, batch_size=16, shuffle=True)
        test_loader = DataLoader(dataset=test_dataset, batch_size=16)

        train_loaders.append(train_loader)
        test_loaders.append(test_loader)

    return train_loaders, test_loaders

def load_data_from_df(df, num_clients: int, test_size: float = 0.2, random_state: int = 42, portion: float = 0.1):
    # Separate the data into fraudulent and non-fraudulent transactions
    df_fraud = df[df['isFraud'] == 1]
    df_non_fraud = df[df['isFraud'] == 0]

    # Check if there are enough fraudulent transactions to distribute
    if len(df_fraud) < num_clients:
        raise ValueError("Not enough fraudulent transactions to distribute to all clients")

    # Randomly shuffle data
    df_fraud = df_fraud.sample(frac=1, random_state=random_state).reset_index(drop=True)
    df_non_fraud = df_non_fraud.sample(frac=1, random_state=random_state).reset_index(drop=True)

    # Split fraudulent transactions evenly across clients
    dfs_fraud = np.array_split(df_fraud, num_clients)
    
    # Split non-fraudulent transactions evenly across clients
    dfs_non_fraud = np.array_split(df_non_fraud, num_clients)

    train_loaders = []
    test_loaders = []

    for i in range(num_clients):
        # Combine the split data ensuring each client gets some fraudulent transactions
        df_combined = pd.concat([dfs_fraud[i], dfs_non_fraud[i]])
        
        # Sample a portion if specified
        if portion < 1.0:
            df_sampled = df_combined.sample(frac=portion, random_state=random_state)
        else:
            df_sampled = df_combined

        # Include more features
        X = df_sampled[['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest', 
                        'changebalanceOrig', 'changebalanceDest', 'hour']].values
        y = df_sampled['isFraud'].values

        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, random_state=random_state)

        # Convert data to PyTorch tensors and create DataLoader objects
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train)
        X_test_tensor = torch.FloatTensor(X_test)
        y_test_tensor = torch.FloatTensor(y_test)

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        train_loader = DataLoader(dataset=train_dataset, batch_size=16, shuffle=True)
        test_loader = DataLoader(dataset=test_dataset, batch_size=16)

        train_loaders.append(train_loader)
        test_loaders.append(test_loader)

    return train_loaders, test_loaders

def disable_progress_bar():
    import os
    os.environ['TQDM_DISABLE'] = '1'