"""Turn raw documents into overlapping text chunks ready for embedding."""

from pathlib import Path

from pypdf import PdfReader

SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}


def extract_text(path: Path) -> str:
    """Pull raw text out of a single document, regardless of format."""
    if path.suffix.lower() == ".pdf":
        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    return path.read_text(encoding="utf-8", errors="ignore")


def chunk_text(text: str, chunk_size: int = 300, overlap: int = 50) -> list[str]:
    """Split text into overlapping word-based chunks.

    Word-based (not token-based) chunking keeps this dependency-free and easy
    to read. The overlap means neighboring chunks share some words, so an
    answer that straddles a chunk boundary doesn't get cut in half.
    """
    words = text.split()
    if not words:
        return []

    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def load_documents(docs_dir: Path) -> list[dict]:
    """Read every supported file in docs_dir and chunk it.

    Returns a list of {"text": chunk, "source": filename} records.
    """
    records = []
    for path in sorted(docs_dir.iterdir()):
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        text = extract_text(path)
        for chunk in chunk_text(text):
            records.append({"text": chunk, "source": path.name})
    return records
