"""Turn raw documents into overlapping text chunks ready for embedding."""

import re
from pathlib import Path

from pypdf import PdfReader

SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}

# Splits after sentence-ending punctuation, only when followed by whitespace
# and then a capital letter/digit/quote - a reasonable heuristic without
# pulling in a full NLP library. It will occasionally over-split on
# abbreviations (e.g. "Dr. Smith") - acceptable for this project; a proper
# sentence tokenizer (e.g. nltk/spacy) would be the fix if that matters later.
_SENTENCE_SPLIT_RE = re.compile(r'(?<=[.!?])\s+(?=[A-Z0-9"\'(])')


def extract_text(path: Path) -> str:
    """Pull raw text out of a single document, regardless of format."""
    if path.suffix.lower() == ".pdf":
        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    return path.read_text(encoding="utf-8", errors="ignore")


def split_sentences(text: str) -> list[str]:
    """Break text into sentences, collapsing whitespace/line breaks first.

    Collapsing whitespace matters most for PDFs, whose extracted text often
    has stray line breaks in the middle of sentences.
    """
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    return _SENTENCE_SPLIT_RE.split(text)


def chunk_text(text: str, max_words: int = 300, overlap_sentences: int = 2) -> list[str]:
    """Group sentences into chunks, never cutting a sentence in half.

    Sentences are packed in order until adding the next one would exceed
    max_words, then a new chunk starts. The last `overlap_sentences` of the
    closing chunk are carried into the next one, so a question whose answer
    straddles a chunk boundary still has enough surrounding context.
    """
    sentences = split_sentences(text)
    if not sentences:
        return []

    chunks: list[str] = []
    current: list[str] = []
    current_words = 0

    def flush():
        if current:
            chunks.append(" ".join(current))

    for sentence in sentences:
        sentence_words = len(sentence.split())
        if current and current_words + sentence_words > max_words:
            flush()
            # Carry the tail of the closing chunk forward for continuity.
            current = current[-overlap_sentences:] if overlap_sentences else []
            current_words = sum(len(s.split()) for s in current)
        current.append(sentence)
        current_words += sentence_words

    flush()
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
