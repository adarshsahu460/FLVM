import subprocess
import time
import os
import requests
import sys
import torch  # We'll detect number of GPUs available

NUM_CLIENTS = 3
NUM_ROUNDS = 2
CLIENT_DATA_DIR = "preprocessed_data"
SERVER_URL = "http://localhost:8080"
API_KEY = os.getenv("API_KEY", "myflkey123")

# Detect available GPUs
NUM_GPUS = torch.cuda.device_count()
print(f"Detected {NUM_GPUS} GPU(s)")

def run_client(cid, round_num, assigned_gpu):
    env = os.environ.copy()
    # Assign client to a specific GPU (or let it share if GPUs < clients)
    env["CUDA_VISIBLE_DEVICES"] = str(assigned_gpu)
    data_dir = f"preprocessed_data/client_{cid}"
    cmd = [
        sys.executable, "client.py",
        "--cid", str(cid),
        "--data-dir", data_dir,
        "--round", str(round_num)
    ]
    return subprocess.Popen(cmd, env=env)

def aggregate():
    subprocess.run([sys.executable, "aggregate.py"], check=True)

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
        # Round-robin GPU assignment
        assigned_gpu = cid % NUM_GPUS if NUM_GPUS > 0 else -1
        print(f"Launching Client {cid} on GPU {assigned_gpu}")
        procs.append(run_client(cid, round_num, assigned_gpu))
        time.sleep(1)  # Optional slight stagger

    # Wait for all clients to finish
    for proc in procs:
        proc.wait()

    aggregate()
    validate()
    time.sleep(2)