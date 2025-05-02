import re
import matplotlib.pyplot as plt

log_file = "./server.log"

# Patterns to extract relevant information
val_pattern = re.compile(r"Validation Loss: ([\d\.]+), Accuracy: ([\d\.]+)%")
client_pattern = re.compile(r"Received upload-weights request from client (\d+) \(dataset_size=(\d+)\)")

# Lists to store extracted data
val_losses = []
val_accuracies = []
val_rounds = []

client_sizes = {}  # {round: {client_id: size}}
current_round = 0

with open(log_file, "r") as f:
    for line in f:
        # Validation metrics
        val_match = val_pattern.search(line)
        if val_match:
            loss = float(val_match.group(1))
            acc = float(val_match.group(2))
            val_losses.append(loss)
            val_accuracies.append(acc)
            val_rounds.append(current_round)
            current_round += 1
        # Client dataset sizes (optional)
        client_match = client_pattern.search(line)
        if client_match:
            client_id = int(client_match.group(1))
            size = int(client_match.group(2))
            if current_round not in client_sizes:
                client_sizes[current_round] = {}
            client_sizes[current_round][client_id] = size

# Plotting Validation Loss and Accuracy
fig, ax1 = plt.subplots(figsize=(10, 6))

color = 'tab:red'
ax1.set_xlabel('Round')
ax1.set_ylabel('Validation Loss', color=color)
ax1.plot(val_rounds, val_losses, color=color, marker='o', label='Validation Loss')
ax1.tick_params(axis='y', labelcolor=color)
ax1.legend(loc='upper left')

ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('Validation Accuracy (%)', color=color)
ax2.plot(val_rounds, val_accuracies, color=color, marker='x', label='Validation Accuracy')
ax2.tick_params(axis='y', labelcolor=color)
ax2.legend(loc='upper right')

plt.title('Federated Learning: Validation Loss and Accuracy per Round')
plt.tight_layout()
plt.show()

# (Optional) Plot client dataset sizes per round
# Uncomment below to plot client sizes
# import pandas as pd
# client_df = pd.DataFrame(client_sizes).T.sort_index()
# client_df.plot(marker='o', figsize=(10, 6))
# plt.title('Client Dataset Sizes per Round')
# plt.xlabel('Round')
# plt.ylabel('Dataset Size')
# plt.legend(title='Client ID')
# plt.tight_layout()
# plt.show()