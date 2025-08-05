from langchain.text_splitter import RecursiveCharacterTextSplitter

def split_into_chunks(pages, chunk_size=1000, chunk_overlap=200):
    """
    Splits extracted PDF text into overlapping chunks for LLM processing.
    Each page remains individually chunked.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    all_chunks = []

    for page in pages:
        text = page["text"]
        metadata = page["metadata"]

        chunks = splitter.split_text(text)

        for i, chunk in enumerate(chunks):
            chunk_metadata = {
                "source": metadata["source"],
                "page": metadata["page"],
                "chunk": i
            }
            all_chunks.append({
                "text": chunk,
                "metadata": chunk_metadata
            })

    return all_chunks
