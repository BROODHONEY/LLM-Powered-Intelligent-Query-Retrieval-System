import numpy as np
import faiss
from typing import List, Dict, Tuple
from InstructorEmbedding import INSTRUCTOR

from embedder import get_embedding_model  # should return the Instructor model
from utils import load_pickle

INDEX_PATH = "vector_store/faiss.index"
CHUNKS_PATH = "vector_store/chunks.pkl"

def semantic_search(query: str, faiss_index: faiss.IndexFlatL2, chunks: List[str], top_k: int = 5) -> List[Dict]:
    model = get_embedding_model()
    instruction = "Represent the document for retrieval:"
    query_embedding = model.encode([instruction, query], normalize_embeddings=True)

    scores, indices = faiss_index.search(query_embedding, top_k)

    return [
        {"chunk": chunks[i], "score": float(scores[0][idx])}
        for idx, i in enumerate(indices[0])
    ]
