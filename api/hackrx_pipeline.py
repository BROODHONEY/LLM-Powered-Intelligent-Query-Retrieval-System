from typing import List
from embedder import get_embedding_model
from decision_engines.decision_engine_ollama import evaluate_decision, generate_answers
from utils import download_and_split_pdf
from FAISS.index_builder import build_or_load_faiss
from FAISS.search_faiss import semantic_search
import time

def process_questions(doc_url: str, questions: List[str]) -> List[str]:

    timings = {}

    start = time.perf_counter()
    chunks = download_and_split_pdf(doc_url)
    timings['download_and_split_pdf'] = time.perf_counter() - start
    print(f"download_and_split_pdf: {timings['download_and_split_pdf']:.2f} seconds \n")

    start = time.perf_counter()
    faiss_index, chunks = build_or_load_faiss(chunks)
    timings['build_or_load_faiss'] = time.perf_counter() - start
    print(f"build_or_load_faiss: {timings['build_or_load_faiss']:.2f} seconds \n")

    answers = []
    print("[INFO] Processing questions...")
    print(f"[INFO] Number of questions: {len(questions)}")

    for q in questions:
        print("[INFO] Searching for relevant clauses...")

        start = time.perf_counter()
        retrieved = semantic_search(q, faiss_index, chunks)
        timings['semantic_search'] = time.perf_counter() - start
        print(f"semantic_search: {timings['semantic_search']:.2f} seconds \n")

        top_texts = [r["chunk"] for r in retrieved]
        print("[INFO] Generating answer...")

        start = time.perf_counter()
        answer = generate_answers(question=q, retrieved_clauses=top_texts)
        timings['generate_answers'] = time.perf_counter() - start
        print(f"generate_answers: {timings['generate_answers']:.2f} seconds \n")

        answers.append(answer)
    
    total_time = sum(timings.values())
    timings['total'] = total_time
    print(f"Total processing time: {total_time:.2f} seconds") 
    return answers
