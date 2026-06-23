"""Central configuration for the local RAG ingestion pipeline."""

from pathlib import Path
from datetime import datetime

# Project-local paths. These keep generated data beside the example code.
PACKAGE_DIR = Path(__file__).parent.resolve()
DATA_DIR = PACKAGE_DIR / "data"
CHROMA_DIR = PACKAGE_DIR / "chroma_db"
COLLECTION_NAME = "rag_docs"

# Small local embedding model used by sentence-transformers.
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
EMBED_DIM = 384

# Retrieval chunk size and overlap measured with tiktoken tokens.
CHUNK_SIZE = 500
CHUNK_OVERLAP = 64

# JSON file that stores chunk hashes so re-ingestion can skip duplicates.
INGEST_LOG = PACKAGE_DIR / "ingest_log.json"
