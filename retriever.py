import torch
from torch.nn.functional import cosine_similarity

def retrieve_top_clauses(query_embedding, clause_embeddings, clauses, top_k=3):
    similarities = cosine_similarity(query_embedding, clause_embeddings)
    top_indices = torch.topk(similarities, k=top_k).indices
    return [clauses[i] for i in top_indices]
