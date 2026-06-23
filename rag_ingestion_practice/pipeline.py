#!/usr/bin/env python3
"""
End-to-end RAG ingestion pipeline.

Usage:
    python pipeline.py --source data/sample.txt.txt
    python pipeline.py --source data/                          # all files in dir
    python pipeline.py --source https://example.com            # web page
    python pipeline.py --reindex                               # force re-ingest
"""

import argparse
from pathlib import Path
from config import DATA_DIR
from extract import extract_from_file, extract_from_url
from chunking import chunk_markdown
from embed import embed_chunks
from store import get_or_create_collection, store_chunks


def run_pipeline(
    sources: list[str],
    reindex: bool = False,
    chroma_path: str | None = None,
    ingest_log_path: str | None = None,
) -> int:
    collection = get_or_create_collection(chroma_path=chroma_path)

    if reindex:
        collection.delete(where={"chunk_index": {"$gte": 0}})
        print("  Cleared existing collection (--reindex)")

    total_chunks = 0

    for src in sources:
        src = src.strip()
        if not src:
            continue

        if src.startswith("http://") or src.startswith("https://"):
            print(f"\n[EXTRACT] URL: {src}")
            doc = extract_from_url(src)
        else:
            path = Path(src)
            if path.is_dir():
                files = sorted(
                    p for p in path.rglob("*")
                    if p.suffix.lower() in {".txt", ".md", ".html", ".pdf", ".docx", ".pptx", ".csv", ".json", ".xml"}
                )
                for f in files:
                    total_chunks += _ingest_one(f, collection, reindex, ingest_log_path=ingest_log_path)
                continue
            else:
                print(f"\n[EXTRACT] file: {path.name}")
                doc = extract_from_file(path)

        if doc is None:
            continue

        chunks = chunk_markdown(doc["markdown"], doc["metadata"])
        if not chunks:
            print("  No chunks produced")
            continue

        texts = [c["text"] for c in chunks]
        print(f"  Chunked into {len(chunks)} pieces")

        print("  Generating embeddings...")
        vectors = embed_chunks(texts)

        count = store_chunks(chunks, collection, dedup=not reindex, ingest_log_path=ingest_log_path)
        total_chunks += count

    print(f"\n[DONE] {total_chunks} new chunks ingested")
    return total_chunks


def _ingest_one(file_path: Path, collection, reindex: bool, ingest_log_path: str | None = None) -> int:
    print(f"\n[EXTRACT] file: {file_path.name}")
    doc = extract_from_file(file_path)
    if doc is None:
        return 0

    chunks = chunk_markdown(doc["markdown"], doc["metadata"])
    if not chunks:
        return 0

    texts = [c["text"] for c in chunks]
    print(f"  Chunked into {len(chunks)} pieces")

    print("  Generating embeddings...")
    vectors = embed_chunks(texts)

    return store_chunks(chunks, collection, dedup=not reindex, ingest_log_path=ingest_log_path)


def main():
    parser = argparse.ArgumentParser(description="RAG ingestion pipeline")
    parser.add_argument(
        "--source", "-s",
        nargs="+",
        default=[str(DATA_DIR)],
        help="File paths, directories, or URLs to ingest",
    )
    parser.add_argument(
        "--reindex",
        action="store_true",
        help="Clear collection and re-ingest everything",
    )
    args = parser.parse_args()

    run_pipeline(sources=args.source, reindex=args.reindex)


if __name__ == "__main__":
    main()

