import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from config.settings import SENTENCE_MODEL_ID, DATABASE_PATH

class EmbeddingsModel:
    def __init__(self):
        self.model = SentenceTransformer(SENTENCE_MODEL_ID)
        self.df = pd.read_parquet(DATABASE_PATH, engine="pyarrow")
        self.embeddings_matrix = np.vstack(self.df["embeddings"].values)
    
    def get_similar_responses(self, query, top_k=2):
        query_embedding = self.model.encode(query)
        similarities = cosine_similarity([query_embedding], self.embeddings_matrix)[0]
        self.df["similarity"] = similarities
        return self.df.nlargest(top_k, "similarity")["completion"].tolist()
