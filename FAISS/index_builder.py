import os
import faiss
from typing import List, Tuple
from InstructorEmbedding import INSTRUCTOR
from utils import extract_text_from_pdfs, split_into_chunks, save_pickle, load_pickle
from embedder import get_embedding_model
from tqdm import tqdm
import numpy as np

INDEX_PATH = "vector_store/faiss.index"
CHUNKS_PATH = "vector_store/chunks.pkl"
model = get_embedding_model()


def build_or_load_faiss(chunks) -> Tuple[faiss.IndexFlatL2, List[str]]:
    os.makedirs("vector_store", exist_ok=True)

    if os.path.exists(INDEX_PATH) and os.path.exists(CHUNKS_PATH):
        print("[INFO] Loading existing FAISS index and chunks...")
        index = faiss.read_index(INDEX_PATH)
        chunks = load_pickle(CHUNKS_PATH)
        return index, chunks

    print("[INFO] Building new FAISS index...")

    # Step 4: Generate embeddings
    print("[INFO] Creating embeddings...")
    instruction = "Represent the document for retrieval:"
    embeddings = []

    for chunk in tqdm(chunks, desc="Embedding chunks"):
        embedding = model.encode([[instruction, chunk]], batch_size=16)
        embeddings.append(embedding[0])

    embeddings = np.array(embeddings)

    # Step 5: Create and fill FAISS index
    print("[INFO] Creating FAISS index...")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)

    print("[INFO] Adding embeddings to index...")
    index.add(embeddings)

    # Step 7: Save index and chunks
    print("[INFO] Saving index and chunks to disk...")
    faiss.write_index(index, INDEX_PATH)
    save_pickle(chunks, CHUNKS_PATH)

    print("[DONE] FAISS index built and saved.")

    return index, chunks
