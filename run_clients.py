import subprocess

def main():
    # Define the range of node IDs and client IDs you want to run
    for client_id in range(5):  
        # Run client.py with the specified node ID and client ID
        command = f"python client.py --node-id {client_id} --client_id {client_id}"
        
        # Launch the client script in a subprocess
        subprocess.Popen(command, shell=True)

if __name__ == "__main__":
    main()

