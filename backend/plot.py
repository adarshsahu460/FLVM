import re
import matplotlib.pyplot as plt
import numpy as np
import ast
import seaborn as sns

log_file = "./server.log"

# Patterns for metrics and confusion matrix
val_pattern = re.compile(
    r"Validation Loss: ([\d\.]+), Accuracy: ([\d\.]+)%"
    r", F1: ([\d\.]+), Precision: ([\d\.]+), Recall: ([\d\.]+)"
)
cm_pattern = re.compile(r"Confusion Matrix: (\[.*\])")

val_losses, val_accuracies, val_f1s, val_precisions, val_recalls, val_rounds = [], [], [], [], [], []
confusion_matrices = []
current_round = 0

with open(log_file, "r") as f:
    for line in f:
        val_match = val_pattern.search(line)
        if val_match:
            val_losses.append(float(val_match.group(1)))
            val_accuracies.append(float(val_match.group(2)))
            val_f1s.append(float(val_match.group(3)))
            val_precisions.append(float(val_match.group(4)))
            val_recalls.append(float(val_match.group(5)))
            val_rounds.append(current_round)
            current_round += 1
        cm_match = cm_pattern.search(line)
        if cm_match:
            cm = np.array(ast.literal_eval(cm_match.group(1)))
            confusion_matrices.append(cm)

# Plotting each metric on a separate graph
plt.figure(figsize=(8, 5))
plt.plot(val_rounds, val_losses, marker='o', color='tab:red')
plt.title('Validation Loss per Round')
plt.xlabel('Round')
plt.ylabel('Loss')
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(val_rounds, val_accuracies, marker='x', color='tab:blue')
plt.title('Validation Accuracy per Round')
plt.xlabel('Round')
plt.ylabel('Accuracy (%)')
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(val_rounds, val_f1s, marker='s', color='tab:green')
plt.title('F1 Score per Round')
plt.xlabel('Round')
plt.ylabel('F1 Score')
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(val_rounds, val_precisions, marker='^', color='tab:orange')
plt.title('Precision per Round')
plt.xlabel('Round')
plt.ylabel('Precision')
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(val_rounds, val_recalls, marker='v', color='tab:purple')
plt.title('Recall per Round')
plt.xlabel('Round')
plt.ylabel('Recall')
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot confusion matrix for each round
for i, cm in enumerate(confusion_matrices):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - Round {i}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
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