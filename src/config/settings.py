from pathlib import Path
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv("../../.env")

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent

# API Configuration
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

# Model Configuration
SENTENCE_MODEL_ID = "C:/Users/ylfgh/Documents/Python_Projects/RAG_Seminaire/models/all-mpnet-base-v2"
#SUMMARIZATION_MODEL_ID = "mrm8488/camembert2camembert_shared-finetuned-french-summarization"
LLM_MODEL = "llama3-70b-8192"

# Audio Configuration
TTS_LANGUAGE = "en"

# Database Configuration
DATABASE_PATH = "C:/Users/ylfgh/Documents/Python_Projects/RAG_Seminaire/data/processed/prompt_dict_updated.parquet"