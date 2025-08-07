from typing import List
from embedder import get_embedding_model
from decision_engines.decision_engine_ollama import evaluate_decision, generate_answers
from utils import download_and_split_pdf
from FAISS.index_builder import build_or_load_faiss
from FAISS.search_faiss import semantic_search

def process_questions(doc_url: str, questions: List[str]) -> List[str]:
    chunks = download_and_split_pdf(doc_url)
    faiss_index, chunks = build_or_load_faiss(chunks)

    answers = []
    print("Processing questions...")
    print(f"Number of questions: {len(questions)}")
    for q in questions:
        print("Searching for relevant clauses...")
        retrieved = semantic_search(q, faiss_index, chunks)
        top_texts = [r["chunk"] for r in retrieved]
        print("Generating answer...")
        answer = generate_answers(question=q, retrieved_clauses=top_texts)
        answers.append(answer)
    return answers
