import os
# import zipfile
from kaggle.api.kaggle_api_extended import KaggleApi

# Define paths
PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
DATA_RAW_DIR = os.path.join(PROJECT_DIR, "data", "raw")
KAGGLE_DATASET = "thejas2002/prompt"

def download_and_extract_kaggle_dataset(dataset: str, dest_folder: str):
    """Downloads and extracts a Kaggle dataset to the specified directory."""
    api = KaggleApi()
    api.authenticate()
    
    # Ensure the destination folder exists
    os.makedirs(dest_folder, exist_ok=True)
    
    # Download dataset
    print(f"Downloading {dataset} to {dest_folder}...")
    api.dataset_download_files(dataset, path=dest_folder, unzip=True)
    print("Download complete.")

if __name__ == "__main__":
    download_and_extract_kaggle_dataset(KAGGLE_DATASET, DATA_RAW_DIR)