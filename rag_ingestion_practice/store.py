"""Storage layer: persist chunks in ChromaDB and query them back."""

import json
from pathlib import Path
import chromadb
from chromadb.config import Settings
import numpy as np

from config import CHROMA_DIR, COLLECTION_NAME, INGEST_LOG


def get_or_create_collection(chroma_path: str | None = None, collection_name: str | None = None):
    """Open a persistent ChromaDB collection, creating it if needed."""
    path_str = chroma_path or str(CHROMA_DIR)
    name = collection_name or COLLECTION_NAME
    Path(path_str).mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(
        path=path_str,
        settings=Settings(anonymized_telemetry=False),
    )
    return client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
    )


def _load_known_hashes(ingest_log_path: str | None = None) -> set:
    """Load previously ingested chunk hashes from the dedup log."""
    path = Path(ingest_log_path) if ingest_log_path else INGEST_LOG
    if path.exists():
        with open(path, "r") as f:
            return set(json.load(f))
    return set()


def _save_known_hashes(hashes: set, ingest_log_path: str | None = None):
    """Persist known chunk hashes so future ingests can skip duplicates."""
    path = Path(ingest_log_path) if ingest_log_path else INGEST_LOG
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(sorted(hashes), f)


def store_chunks(chunk_docs: list[dict], collection, dedup: bool = True, ingest_log_path: str | None = None) -> int:
    """Store chunk documents in ChromaDB and return the number newly added."""
    if not chunk_docs:
        return 0

    known = _load_known_hashes(ingest_log_path=ingest_log_path) if dedup else set()
    new_count = 0

    for doc in chunk_docs:
        h = doc["metadata"]["chunk_hash"]
        if h in known:
            continue

        meta = dict(doc["metadata"])
        # Chroma IDs are stable per source and chunk index.
        chunk_id = f"{meta['source']}#{meta['chunk_index']}"

        collection.add(
            ids=[chunk_id],
            documents=[doc["text"]],
            metadatas=[meta],
        )
        known.add(h)
        new_count += 1

    _save_known_hashes(known, ingest_log_path=ingest_log_path)
    print(f"  Stored {new_count} new chunks ({len(chunk_docs) - new_count} duplicates skipped)")
    return new_count


def retrieve(query_text: str, collection, k: int = 5, query_vector: list[float] | None = None) -> list[dict]:
    """Retrieve top-k chunks for a query or precomputed query vector."""
    if query_vector is not None:
        qvec = query_vector
    else:
        from embed import embed_query
        qvec = embed_query(query_text)

    results = collection.query(
        query_embeddings=[qvec],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )

    out = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        out.append({
            "text": doc,
            "metadata": meta,
            # Chroma returns distance; this converts it into similarity-like score.
            "score": 1.0 - dist,
        })
    return out
