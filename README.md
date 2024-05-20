# Anomaly Detection in Federated Learning

This repository pertains to Semester 1 of the Master of Applied IT, focusing on the Federated Learning project. The primary research inquiry revolves around exploring the influence of diverse data sources—such as numerical, textual, and image data—on the accuracy of federated learning models within the realm of anomaly detection. It aims to address the challenges presented by data heterogeneity in this context."

## Instructions
1. Install the dependancies in the `requirements.txt` file.
   
```
  pip install -r requirements.txt
```

2. Start `anomaly.py`
```
python anomaly.py
```

4. Start the server in the terminal as follows:
```
python server.py
```

4. Start Client 1 in the first terminal
```
python client.py --node-id 0
```

6. Start Client 2 in the second terminal
```
python client.py --node-id 0
```

7. Start Client 3 in the third terminal
```
python client.py --node-id 0
```

8. Start Client 4 in the fourth terminal
```
python client.py --node-id 0
```

9. Start Client 5 in the fifth terminal
```
python client.py --node-id 0
```
