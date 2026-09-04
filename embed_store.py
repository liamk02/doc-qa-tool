"""Embedding model + a minimal local vector store (numpy, cosine similarity).

No vector database here on purpose - for a first RAG project, this file is
short enough to read top to bottom and see exactly how "search by meaning"
works under the hood.
"""

import pickle
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"  # small, fast, runs on CPU, free, good enough at this scale

_model = None


def get_model() -> SentenceTransformer:
    """Load the embedding model once and reuse it (downloads ~90MB on first use)."""
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed(texts: list[str]) -> np.ndarray:
    """Turn a list of strings into a matrix of embedding vectors."""
    return get_model().encode(texts, show_progress_bar=False, convert_to_numpy=True)


def save_index(records: list[dict], embeddings: np.ndarray, path: Path) -> None:
    with open(path, "wb") as f:
        pickle.dump({"records": records, "embeddings": embeddings}, f)


def load_index(path: Path) -> tuple[list[dict], np.ndarray]:
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data["records"], data["embeddings"]


def top_k(query_embedding: np.ndarray, embeddings: np.ndarray, k: int = 4) -> list[int]:
    """Return the indices of the k most similar embeddings, via cosine similarity."""
    query_norm = query_embedding / np.linalg.norm(query_embedding)
    matrix_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    scores = matrix_norm @ query_norm
    return np.argsort(scores)[::-1][:k].tolist()
