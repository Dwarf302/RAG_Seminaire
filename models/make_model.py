import os
from sentence_transformers import SentenceTransformer

# Define the directory to save the model
PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
MODEL_DIR = os.path.join(PROJECT_DIR, "models", "all-mpnet-base-v2")

def download_and_save_model(model_name: str, save_path: str):
    """Downloads a SentenceTransformer model and saves it permanently."""
    os.makedirs(save_path, exist_ok=True)  # Ensure directory exists
    model = SentenceTransformer(model_name, cache_folder=save_path)
    model.save(save_path)
    print(f"Model '{model_name}' saved at: {save_path}")

if __name__ == "__main__":
    download_and_save_model("sentence-transformers/all-mpnet-base-v2", MODEL_DIR)
