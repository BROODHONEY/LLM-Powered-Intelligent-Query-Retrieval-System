from fastapi import FastAPI, Query
from FAISS.search_faiss import semantic_search
from FAISS.index_builder import build_or_load_faiss

app = FastAPI()

# Load FAISS index and associated chunks on startup
faiss_index, chunks = build_or_load_faiss()

@app.get("/")
def root():
    return {"message": "Welcome to Semantic PDF Search"}

@app.get("/search")
def search(query: str = Query(..., description="Your semantic query")):
    # Pass both the index and chunks to the search function
    results = semantic_search(query, faiss_index, chunks)
    return {
        "query": query,
        "results": results
    }
