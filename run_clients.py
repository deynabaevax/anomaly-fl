import subprocess

def main():
    # Define the range of node IDs you want to run
    for node_id in range(5):  # Adjust range as needed
        # Construct the command to run client.py with the specified node ID
        command = f"python client.py --node-id {node_id}"
        
        # Launch the client script in a subprocess
        subprocess.Popen(command, shell=True)

if __name__ == "__main__":
    main()
