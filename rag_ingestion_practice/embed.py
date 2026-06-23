"""Embedding layer: convert text into normalized vector embeddings."""

from functools import lru_cache
import numpy as np
from sentence_transformers import SentenceTransformer
from config import EMBED_MODEL_NAME


@lru_cache(maxsize=1)
def _load_model() -> SentenceTransformer:
    """Load the embedding model once and reuse it across calls."""
    print(f"  Loading embedding model: {EMBED_MODEL_NAME}")
    return SentenceTransformer(EMBED_MODEL_NAME)


def embed_chunks(texts: list[str], model: SentenceTransformer | None = None) -> np.ndarray:
    """Embed a batch of chunk texts for storage in the vector database."""
    if model is None:
        model = _load_model()
    return model.encode(texts, show_progress_bar=True, normalize_embeddings=True)


def embed_query(text: str, model: SentenceTransformer | None = None) -> list[float]:
    """Embed a user query and return a plain list for ChromaDB queries."""
    if model is None:
        model = _load_model()
    vec = model.encode([text], normalize_embeddings=True)[0]
    return vec.tolist()
