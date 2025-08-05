from query_parser import parse_query
from embedder import embed_clauses, embed_query
from retriever import retrieve_top_clauses
from decision_engine_gemini import evaluate_decision

def main():
    with open("data/clauses.txt") as f:
        clauses = [line.strip() for line in f.readlines() if line.strip()]
    with open("examples/sample_query.txt") as f:
        query = f.read().strip()
    
    parsed_query = parse_query(query)
    clause_embeddings = embed_clauses(clauses)
    query_embedding = embed_clauses(query)
    top_clauses = retrieve_top_clauses(query_embedding, clause_embeddings, clauses)

    result = evaluate_decision(parsed_query, top_clauses)
    print(result)

if __name__ == "__main__":
    main()