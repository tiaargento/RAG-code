#!/usr/bin/env python3
"""
Query the vector store and print relevant chunks.

Usage:
    python retrieve.py "what is medical report about?"
    python retrieve.py                          # interactive loop
"""

import argparse
import sys
from store import get_or_create_collection, retrieve


def main():
    parser = argparse.ArgumentParser(description="Retrieve chunks from vector store")
    parser.add_argument("query", nargs="*", help="Query text")
    parser.add_argument("-k", type=int, default=5, help="Number of results (default 5)")
    args = parser.parse_args()

    collection = get_or_create_collection()
    count = collection.count()
    print(f"Collection has {count} chunks\n")

    if args.query:
        queries = [" ".join(args.query)]
    else:
        queries = _interactive_loop()

    for q in queries:
        if not q.strip():
            continue
        print(f"\n{'='*60}")
        print(f"Query: {q}")
        print(f"{'='*60}")
        results = retrieve(q, collection, k=args.k)
        if not results:
            print("No results found.")
            continue
        for i, r in enumerate(results, 1):
            src = r["metadata"].get("source", "?")
            title = r["metadata"].get("title", "?")
            chunk_idx = r["metadata"].get("chunk_index", "?")
            print(f"\n--- Result {i} (score: {r['score']:.3f}) ---")
            print(f"  source: {src} | title: {title} | chunk: {chunk_idx}")
            print(f"  text: {r['text'][:300]}...")


def _interactive_loop():
    print("Enter queries (empty line to quit):")
    queries = []
    try:
        while True:
            line = input("> ").strip()
            if not line:
                break
            queries.append(line)
    except (EOFError, KeyboardInterrupt):
        pass
    return queries


if __name__ == "__main__":
    main()
