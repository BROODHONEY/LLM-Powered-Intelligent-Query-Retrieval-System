from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

def embed_clauses(clauses):
    return model.encode(clauses, convert_to_tensor=True)

def embed_query(query):
    return model.encode([query], convert_to_tensor=True)[0]