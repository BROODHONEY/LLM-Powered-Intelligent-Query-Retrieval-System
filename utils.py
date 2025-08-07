import os
import pickle
import fitz  # PyMuPDF
import tempfile
import requests
from typing import List

def extract_text_from_pdfs(directory: str) -> str:
    all_text = ""
    for filename in os.listdir(directory):
        if filename.endswith(".pdf"):
            with fitz.open(os.path.join(directory, filename)) as doc:
                for page in doc:
                    all_text += page.get_text()
    return all_text

def download_and_split_pdf(pdf_url: str, chunk_size: int = 300) -> List[str]:
    """Download a PDF from a URL and split it into text chunks."""
    response = requests.get(pdf_url)
    response.raise_for_status()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(response.content)
        tmp_path = tmp_file.name

    doc = fitz.open(tmp_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    doc.close()
    os.remove(tmp_path)

    return split_into_chunks(full_text)

def split_into_chunks(text: str, chunk_size: int = 500, overlap: int = 50) -> list:
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

def save_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)
