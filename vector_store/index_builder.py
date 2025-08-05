from InstructorEmbedding import INSTRUCTOR
import faiss
import os
import pickle

embedding_model = INSTRUCTOR('hkunlp/instructor-xl')

def get_embeddings(chunks):
    """
    Converts text chunks to embedding vectors.
    """
    instructions = ["Represent the insurance policy sentence for retrieval:"] * len(chunks)
    texts = [chunk["text"] for chunk in chunks]
    embeddings = embedding_model.encode(list(zip(instructions, texts)))
    return embeddings

def build_vector_store(chunks, save_path="vector_store/index.faiss", metadata_path="vector_store/metadata.pkl"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Get vectors
    embeddings = get_embeddings(chunks)

    # Store in FAISS
    dim = embeddings[0].shape[0]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    # Save index and metadata
    faiss.write_index(index, save_path)
    with open(metadata_path, "wb") as f:
        pickle.dump(chunks, f)

    print(f"✅ Stored {len(chunks)} vectors in FAISS.")