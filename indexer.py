"""Build (or rebuild) the search index from documents in a folder.

Shared by app.py's `ingest` command (manual, local use) and web.py (automatic,
on server startup - so a fresh deploy needs no manual step).
"""

from pathlib import Path

from chunking import load_documents
from embed_store import backend_name, embed, index_backend, load_index, save_index


def build_index(docs_dir: Path, index_path: Path) -> int:
    """Chunk + embed every document in docs_dir and save the index.

    Returns the number of chunks indexed (0 if docs_dir had no supported files).
    """
    records = load_documents(docs_dir)
    if not records:
        return 0
    embeddings = embed([r["text"] for r in records], input_type="document")
    save_index(records, embeddings, index_path)
    return len(records)


def load_or_build_index(docs_dir: Path, index_path: Path) -> tuple[list[dict], object]:
    """Load the index, rebuilding it first if it's missing OR was built by a
    different embedding backend (e.g. you set/unset VOYAGE_API_KEY since the
    last build - vectors from different backends aren't compatible).
    """
    stale = index_path.exists() and index_backend(index_path) != backend_name()
    if not index_path.exists() or stale:
        build_index(docs_dir, index_path)
    return load_index(index_path)
