#!/usr/bin/env python3
"""Query the vector store and optionally generate a grounded answer.

Usage:
    python retrieve.py "what is medical report about?"
    python retrieve.py "what is medical report about?" --generate
    python retrieve.py                          # interactive loop
"""

import argparse
from generate import DEFAULT_GENERATION_MODEL, generate_answer
from store import get_or_create_collection, retrieve


def main():
    """Parse CLI arguments, retrieve chunks, and optionally synthesize an answer."""
    parser = argparse.ArgumentParser(description="Retrieve chunks from vector store")
    parser.add_argument("query", nargs="*", help="Query text")
    parser.add_argument("-k", type=int, default=5, help="Number of results (default 5)")
    parser.add_argument(
        "-g", "--generate",
        action="store_true",
        help="Use an Ollama LLM to synthesize an answer from retrieved chunks",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_GENERATION_MODEL,
        help=f"Ollama model for --generate (default {DEFAULT_GENERATION_MODEL})",
    )
    parser.add_argument(
        "--show-chunks",
        action="store_true",
        help="With --generate, also print raw retrieved chunks",
    )
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
        if args.generate:
            # Generation is optional; retrieval still works without Ollama installed/running.
            try:
                answer = generate_answer(q, results, model=args.model)
                print("\nAnswer:")
                print(answer)
                print("\nSources:")
                for i, r in enumerate(results, 1):
                    src = r["metadata"].get("source", "?")
                    title = r["metadata"].get("title", "?")
                    print(f"  {i}. {title} ({r['score']:.3f}) - {src}")
            except RuntimeError as exc:
                print(f"\nGeneration unavailable: {exc}")

            if not args.show_chunks:
                # In answer mode, hide raw chunks unless the user asks to inspect them.
                continue

        for i, r in enumerate(results, 1):
            src = r["metadata"].get("source", "?")
            title = r["metadata"].get("title", "?")
            chunk_idx = r["metadata"].get("chunk_index", "?")
            print(f"\n--- Result {i} (score: {r['score']:.3f}) ---")
            print(f"  source: {src} | title: {title} | chunk: {chunk_idx}")
            print(f"  text: {r['text'][:300]}...")


def _interactive_loop():
    """Read multiple queries from stdin until the user submits an empty line."""
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
