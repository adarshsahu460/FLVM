import os
import kagglehub
import shutil
from dotenv import load_dotenv, set_key

# Set KAGGLE_CONFIG_DIR to the project directory (where kaggle.json is located)
project_dir = os.path.abspath(os.path.dirname(__file__))
os.environ["KAGGLE_CONFIG_DIR"] = project_dir

# Path to .env file
env_file = os.path.join(project_dir, ".env")

# Create .env file with empty dataset_path if it doesn't exist
if not os.path.exists(env_file):
    with open(env_file, "w") as f:
        f.write("dataset_path=\n")

# Load environment variables
load_dotenv()

# Download dataset
dataset_path = kagglehub.dataset_download("abishekdaskhna/oasis-alzheimers-detection")
print("Path to dataset files:", dataset_path)

# Update dataset_path in .env file
set_key(env_file, "dataset_path", dataset_path)
print("Updated .env with dataset_path:", dataset_path)

# Copy the 'data' folder to the current project directory
data_source_path = os.path.join(dataset_path, "data")
data_dest_path = os.path.join(project_dir, "data")

try:
    if os.path.exists(data_source_path):
        shutil.copytree(data_source_path, data_dest_path, dirs_exist_ok=True)
        print(f"Copied 'data' folder to: {data_dest_path}")
    else:
        print(f"Warning: 'data' folder not found in {dataset_path}")
except Exception as e:
    print(f"Error copying 'data' folder: {str(e)}")