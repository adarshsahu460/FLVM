import torch
import transformers
import flwr as fl
import nibabel
import numpy as np
from PIL import Image

print("PyTorch version:", torch.__version__)
print("Transformers version:", transformers.__version__)
print("Flower version:", fl.__version__)
print("Nibabel version:", nibabel.__version__)
print("NumPy version:", np.__version__)
print("PIL version:", Image.__version__)
print("CUDA available:", torch.cuda.is_available())