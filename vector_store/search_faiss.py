import faiss
import pickle
import numpy as np
from InstructorEmbedding import INSTRUCTOR

embedding_model = INSTRUCTOR("hkunlp/instructor-xl")

def embed_query(query: str):
    instruction = "Represent the candidate query for retrieval:"
    return embedding_model.encode([[instruction, query]])[0]

def search_faiss_index(query: str, index_path="vector_store/index.faiss", metadata_path="vector_store/metadata.pkl", top_k=5):
    # Load index and metadata
    index = faiss.read_index(index_path)
    with open(metadata_path, "rb") as f:
        metadata = pickle.load(f)

    # Embed query
    query_vector = embed_query(query).astype("float32")

    # Search
    distances, indices = index.search(np.array([query_vector]), top_k)

    # Get results
    results = []
    for i in indices[0]:
        if i < len(metadata):
            results.append(metadata[i])

    return results