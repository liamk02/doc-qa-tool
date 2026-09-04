"""Build (or rebuild) the search index from documents in a folder.

Shared by app.py's `ingest` command (manual, local use) and web.py (automatic,
on server startup - so a fresh deploy needs no manual step).
"""

from pathlib import Path

from chunking import load_documents
from embed_store import embed, save_index


def build_index(docs_dir: Path, index_path: Path) -> int:
    """Chunk + embed every document in docs_dir and save the index.

    Returns the number of chunks indexed (0 if docs_dir had no supported files).
    """
    records = load_documents(docs_dir)
    if not records:
        return 0
    embeddings = embed([r["text"] for r in records])
    save_index(records, embeddings, index_path)
    return len(records)
