# Anomaly Detection in Federated Learning

Welcome to the repository for the Federated Machine Learning project undertaken during Semester 1 of the Master of Applied IT program. This project focuses on exploring the impact of data heterogeneity on anomaly detection within federated learning environments.
# Getting Started
To get started with this project, follow these instructions:

## Prerequisites
Make sure you have Python installed on your system. This project uses dependencies listed in requirements.txt.

## Installation 
Install the dependancies by running:
   
```
  pip install -r requirements.txt
```
## Running the Server

Start the server by executing the following command in your terminal:
```
python server.py
```

## Starting Clients
Start the clients by executing the following command in your terminal:
```
python run_clients.py
```


# Project Structure
This experiment consists of 4 Python files - `anomaly.py`, `client.py`, `server.py`, and `run_clients.py`. 

- *anomaly.py* - Contains all the functionalities related to data loading, model definition, training and evaluation
- *client.py* - Contains the client-side logic for FL. It utilises the methods defined in anomaly.py to train and evaluate the model on the local data. 
- *server.py* - Contains the server logic with a specified number of clients, rounds, and aggregation strategy. 
- *run_clients.py* - Contains the logic for automating the execution of `client.py` across multiple instances, each initialised with a unique `client_id`. This script uses the subprocess module to launch these instances concurrently instead of running the `client.py` code multiple times in the terminal.

