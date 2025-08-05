import fitz  # PyMuPDF
import os

def load_pdf_text(file_path):
    """
    Extracts text from a PDF file and returns a list of page-level chunks with metadata.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} does not exist.")

    doc = fitz.open(file_path)
    all_chunks = []

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text().strip()

        if text:
            all_chunks.append({
                "text": text,
                "metadata": {
                    "source": os.path.basename(file_path),
                    "page": page_num + 1
                }
            })

    doc.close()
    return all_chunks
