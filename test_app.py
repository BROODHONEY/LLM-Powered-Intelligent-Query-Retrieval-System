from FAISS.index_builder import build_or_load_faiss
from FAISS.search_faiss import semantic_search
from decision_engines.decision_engine_ollama import evaluate_decision
from query_parser import parse_query, build_search_input

def main():
    # Step 1: Build or load FAISS index and chunks
    index, chunks = build_or_load_faiss()

    # Step 2: Ask a test query
    query = input("Enter your search query: ")
    parsed_query = parse_query(query)
    search_sentence = build_search_input(parsed_query)

    # Step 3: Perform semantic search
    results = semantic_search(search_sentence, index, chunks)

    # Step 4: Display results
    # print("\nTop Results:\n" + "-"*40)
    # for i, res in enumerate(results, 1):
    #     print(f"\nResult {i}:")
    #     print(f"Score: {res['score']:.4f}")
    #     print(f"Chunk:\n{res['chunk']}\n")

    decision = evaluate_decision(parsed_query=parsed_query, retrieved_clauses=results)
    print("\nDecision Engine Output:")
    print(decision)

if __name__ == "__main__":
    main()
