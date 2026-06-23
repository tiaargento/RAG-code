from functools import lru_cache
import numpy as np
from sentence_transformers import SentenceTransformer
from config import EMBED_MODEL_NAME


@lru_cache(maxsize=1)
def _load_model() -> SentenceTransformer:
    print(f"  Loading embedding model: {EMBED_MODEL_NAME}")
    return SentenceTransformer(EMBED_MODEL_NAME)


def embed_chunks(texts: list[str], model: SentenceTransformer | None = None) -> np.ndarray:
    if model is None:
        model = _load_model()
    return model.encode(texts, show_progress_bar=True, normalize_embeddings=True)


def embed_query(text: str, model: SentenceTransformer | None = None) -> list[float]:
    if model is None:
        model = _load_model()
    vec = model.encode([text], normalize_embeddings=True)[0]
    return vec.tolist()
