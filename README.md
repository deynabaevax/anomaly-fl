# Anomaly Detection in Federated Learning

# 📘 Project Overview

This project investigates how data heterogeneity affects anomaly detection in a Federated Learning (FL) environment. Using the Isolation Forest (iForest) model, the project simulates distributed training across multiple clients on financial transaction data to detect fraudulent behavior — without centralizing sensitive data.

The research was conducted as part of the Master's in Applied IT at Fontys University of Applied Sciences, and aims to contribute to more secure and privacy-preserving anomaly detection systems in domains like finance and cybersecurity.

## 📄 Project Report

Read the full research paper here:

[Exploring Federated Learning for Anomaly Detection.pdf](https://github.com/user-attachments/files/21368388/Exploring.Federated.Learning.for.Anomaly.Detection.pdf)


# ⚙️ Key Features

- Federated Learning Simulation
   - Implements a client-server architecture using Flower to simulate decentralized learning.

- Anomaly Detection
   - Applies machine learning models (iForest, LOF, OCSVM) to detect outliers in distributed datasets.

- Data Heterogeneity Analysis
   - Compares the effects of even vs uneven fraud label distribution across clients on model performance.

- Data Augmentation & Balancing
   - Uses SMOTE to address extreme class imbalance (0.13% fraud cases).

- Scalability Testing
   - Evaluates how performance varies with multiple clients and complex data partitions.

# 🛠️ Technologies Used

- Python – Core programming language

- Flower – Federated learning framework

- Scikit-learn – Isolation Forest, LOF, and One-Class SVM

- SMOTE – For class balancing

- Pandas, NumPy – Data manipulation

- Matplotlib, Seaborn – Visual analytics

- Jupyter Notebook – EDA and reporting

# 📂 Repository Structure

This experiment consists of 4 Python files - `anomaly.py`, `client.py`, `server.py`, and `run_clients.py`. 

- *anomaly.py* - Contains all the functionalities related to data loading, model definition, training and evaluation
- *client.py* - Contains the client-side logic for FL. It utilises the methods defined in anomaly.py to train and evaluate the model on the local data. 
- *server.py* - Contains the server logic with a specified number of clients, rounds, and aggregation strategy. 
- *run_clients.py* - Contains the logic for automating the execution of `client.py` across multiple instances, each initialised with a unique `client_id`. This script uses the subprocess module to launch these instances concurrently instead of running the `client.py` code multiple times in the terminal.

# 🚀 Getting Started

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

# 📊 Experimental Insights

- Even Fraud Distribution: Led to instability and fluctuating performance across clients due to varied learning challenges.

- Uneven Fraud Distribution: Resulted in more stable, predictable training and faster convergence.

- Baseline Comparison: One-Class SVM had highest accuracy (0.85), while iForest achieved perfect recall (1.0) but lower precision.

- Federated vs Centralized: FL maintained privacy but required careful data partitioning for optimal accuracy.

# 🎯 Outcomes

- Demonstrated the feasibility of anomaly detection in federated systems.

- Highlighted the impact of data heterogeneity on model accuracy and convergence.

- Provided practical guidelines for partitioning strategies and FL architecture design.

- Offered a reusable simulation framework for future FL experiments in real-world anomaly detection tasks.

