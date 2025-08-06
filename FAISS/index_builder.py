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

def build_or_load_faiss() -> Tuple[faiss.IndexFlatL2, List[str]]:
    os.makedirs("vector_store", exist_ok=True)

    if os.path.exists(INDEX_PATH) and os.path.exists(CHUNKS_PATH):
        print("[INFO] Loading existing FAISS index and chunks...")
        index = faiss.read_index(INDEX_PATH)
        chunks = load_pickle(CHUNKS_PATH)
        return index, chunks

    print("[INFO] Building new FAISS index...")

    # Step 1: Extract text
    print("[STEP 1] Extracting text from PDFs...")
    texts = extract_text_from_pdfs("data")
    print(f"[INFO] Extracted {len(texts)} documents.")

    # Step 2: Split into chunks
    print("[STEP 2] Splitting documents into chunks...")
    chunks = split_into_chunks(texts)
    print(f"[INFO] Split into {len(chunks)} chunks.")

    # Step 3: Get embedding model
    print("[STEP 3] Loading embedding model...")
    model = get_embedding_model()

    # Step 4: Generate embeddings
    print("[STEP 4] Creating embeddings...")
    instruction = "Represent the document for retrieval:"
    embeddings = []

    for chunk in tqdm(chunks, desc="Embedding chunks"):
        embedding = model.encode([[instruction, chunk]])
        embeddings.append(embedding[0])

    embeddings = np.array(embeddings)

    # Step 5: Create and fill FAISS index
    print("[STEP 5] Creating FAISS index...")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)

    print("[STEP 6] Adding embeddings to index...")
    index.add(embeddings)

    # Step 7: Save index and chunks
    print("[STEP 7] Saving index and chunks to disk...")
    faiss.write_index(index, INDEX_PATH)
    save_pickle(chunks, CHUNKS_PATH)

    print("[DONE] FAISS index built and saved.")

    return index, chunks
