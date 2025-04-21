import subprocess
import time
import os
import requests
import sys  # Ensure we use the venv Python

NUM_CLIENTS = 5
NUM_ROUNDS = 20
CLIENT_DATA_DIR = "preprocessed_data"
SERVER_URL = "http://localhost:8080"
API_KEY = os.getenv("API_KEY", "myflkey123")

def run_client(cid, round_num):
    cmd = [
        sys.executable, "client.py",  # Use the current Python interpreter
        "--cid", str(cid),
        "--data-dir", CLIENT_DATA_DIR,
        "--round", str(round_num)
    ]
    return subprocess.Popen(cmd)

def aggregate():
    subprocess.run([sys.executable, "aggregate.py"], check=True)  # Use the current Python interpreter

def validate():
    resp = requests.get(
        f"{SERVER_URL}/validate-global-model",
        headers={"X-API-Key": API_KEY}
    )
    print("Validation:", resp.json())

for round_num in range(1, NUM_ROUNDS + 1):
    print(f"\n=== Federated Round {round_num} ===")
    procs = []
    for cid in range(NUM_CLIENTS):
        procs.append(run_client(cid, round_num))
        time.sleep(2)  # Add a 2-second delay between starting each client
    for proc in procs:
        proc.wait()
    aggregate()
    validate()
    time.sleep(2)