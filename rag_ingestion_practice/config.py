from pathlib import Path
from datetime import datetime

PACKAGE_DIR = Path(__file__).parent.resolve()
DATA_DIR = PACKAGE_DIR / "data"
CHROMA_DIR = PACKAGE_DIR / "chroma_db"
COLLECTION_NAME = "rag_docs"

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
EMBED_DIM = 384

CHUNK_SIZE = 500
CHUNK_OVERLAP = 64

INGEST_LOG = PACKAGE_DIR / "ingest_log.json"
