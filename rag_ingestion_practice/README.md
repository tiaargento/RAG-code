# RAG Ingestion Pipeline

A modular, local-first RAG ingestion pipeline using MarkItDown, sentence-transformers, and ChromaDB.

## Architecture

```
Source (files / web pages)
  → [extract.py]  MarkItDown → clean Markdown + metadata
  → [chunk.py]    heading-aware + token-aware splitting (~500 tok)
  → [embed.py]    sentence-transformers (all-MiniLM-L6-v2)
  → [store.py]    ChromaDB (persistent, local, cosine similarity)
  → [retrieve.py] query → embed → top-k chunks
```

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Ingest sample files
python pipeline.py --source data/

# Query
python retrieve.py "what is this document about?"
```

## Modules

| Module | Role |
|---|---|
| `config.py` | Paths, model name, chunk parameters |
| `extract.py` | File & URL → Markdown via MarkItDown |
| `chunk.py` | Heading-aware split + token cap (500 tok, 64 overlap) |
| `embed.py` | SentenceTransformer model (cached) |
| `store.py` | ChromaDB upsert with SHA256 dedup |
| `pipeline.py` | CLI orchestration: `--source` + `--reindex` |
| `retrieve.py` | Query loop: `-k 5 "your question"` |

## Deduplication

Chunk hashes (SHA256, first 16 hex chars) are stored in `ingest_log.json`.
On subsequent runs, chunks with existing hashes are skipped.

Use `--reindex` to clear the store and re-ingest everything.

## Tests

```bash
# Run all tests (unit + integration)
pytest tests/ -v

# Skip the real model integration test (takes ~10s first time)
pytest tests/ -v -m "not integration"

# Run only the integration test
pytest tests/ -v -m integration
```

31 tests covering: file extraction, HTML extraction, empty/error handling,
heading-aware chunking, token-aware splitting, metadata correctness,
embedding shape and normalization (mocked + real model), Chroma roundtrip,
deduplication with persisted hash log, retrieval with query vectors,
and end-to-end pipeline smoke tests with temp directories.

## Metadata

Each chunk stores: `source`, `title`, `doc_type`, `chunk_index`, `timestamp`, `chunk_hash`.

## Production Notes

- **Embedding model**: swap `all-MiniLM-L6-v2` for `text-embedding-3-small` (OpenAI) or `BAAI/bge-large-en-v1.5` for higher quality.
- **Vector store**: ChromaDB works for <1M chunks. Beyond that, consider Qdrant, Weaviate, or pgvector.
- **Async**: wrap file I/O and HTTP calls with `asyncio` for parallel ingestion.
- **Chunking**: test different sizes (256–1024) per your use case. Add overlap to avoid splitting context.
- **Metadata filtering**: ChromaDB supports `where` filters on metadata at query time.
