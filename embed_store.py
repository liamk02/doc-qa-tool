"""Embeddings + a minimal local vector store (numpy, cosine similarity).

Two embedding backends, chosen automatically:
- Local (sentence-transformers): the default. Free, no extra API key, runs
  entirely on your machine - but loading it pulls in PyTorch, which is too
  memory-heavy for a small hosted server (this is what ran a Render free
  instance, 512MB RAM, out of memory).
- Voyage AI (hosted API): used automatically whenever a VOYAGE_API_KEY is
  set. No PyTorch, tiny memory footprint - this is what the public
  deployment uses. Both `sentence-transformers` import AND its call happen
  lazily, inside the local-only branch, so a process that never needs it
  (e.g. Render, with VOYAGE_API_KEY set) never loads PyTorch into memory at all.

No vector database here on purpose - for a first RAG project, the search
itself (top_k, below) is short enough to read top to bottom and see exactly
how "search by meaning" works under the hood, whichever backend made the vectors.
"""

import os
import pickle
from pathlib import Path

import numpy as np

LOCAL_MODEL_NAME = "all-MiniLM-L6-v2"
VOYAGE_MODEL_NAME = "voyage-4-lite"

_local_model = None
_voyage_client = None


def backend_name() -> str:
    """Which backend is active, decided by whether VOYAGE_API_KEY is set."""
    return "voyage" if os.environ.get("VOYAGE_API_KEY") else "local"


def _get_local_model():
    global _local_model
    if _local_model is None:
        from sentence_transformers import SentenceTransformer  # lazy - see module docstring
        _local_model = SentenceTransformer(LOCAL_MODEL_NAME)
    return _local_model


def _get_voyage_client():
    global _voyage_client
    if _voyage_client is None:
        import voyageai
        _voyage_client = voyageai.Client()  # reads VOYAGE_API_KEY from env
    return _voyage_client


def embed(texts: list[str], input_type: str | None = None) -> np.ndarray:
    """Turn a list of strings into a matrix of embedding vectors.

    `input_type` ("document" when indexing, "query" when embedding a
    question) lets Voyage tune the vectors for retrieval; the local model
    ignores it.
    """
    if backend_name() == "voyage":
        result = _get_voyage_client().embed(texts, model=VOYAGE_MODEL_NAME, input_type=input_type)
        return np.array(result.embeddings)
    return _get_local_model().encode(texts, show_progress_bar=False, convert_to_numpy=True)


def save_index(records: list[dict], embeddings: np.ndarray, path: Path) -> None:
    # Tag the index with the backend that built it - vectors from different
    # backends have different dimensions/meaning and can't be mixed.
    with open(path, "wb") as f:
        pickle.dump({"records": records, "embeddings": embeddings, "backend": backend_name()}, f)


def load_index(path: Path) -> tuple[list[dict], np.ndarray]:
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data["records"], data["embeddings"]


def index_backend(path: Path) -> str | None:
    """Which backend built the index on disk, or None if the file predates this field."""
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data.get("backend")


def top_k(query_embedding: np.ndarray, embeddings: np.ndarray, k: int = 4) -> list[int]:
    """Return the indices of the k most similar embeddings, via cosine similarity."""
    query_norm = query_embedding / np.linalg.norm(query_embedding)
    matrix_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    scores = matrix_norm @ query_norm
    return np.argsort(scores)[::-1][:k].tolist()
