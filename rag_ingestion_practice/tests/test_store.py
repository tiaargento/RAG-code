import numpy as np
import pytest

from store import get_or_create_collection, store_chunks, retrieve


class TestStoreChunks:
    def test_store_and_count(self, chroma_path, chroma_collection, chunk_docs):
        count = store_chunks(chunk_docs, chroma_collection, dedup=False)
        assert count == len(chunk_docs)
        assert chroma_collection.count() == len(chunk_docs)

    def test_dedup_skips_exact_duplicates(self, chroma_path, chroma_collection, chunk_docs, ingest_log_path):
        first = store_chunks(chunk_docs, chroma_collection, dedup=True, ingest_log_path=ingest_log_path)
        second = store_chunks(chunk_docs, chroma_collection, dedup=True, ingest_log_path=ingest_log_path)
        assert first == len(chunk_docs)
        assert second == 0
        assert chroma_collection.count() == len(chunk_docs)

    def test_empty_chunks(self, chroma_collection):
        count = store_chunks([], chroma_collection)
        assert count == 0

    def test_dedup_log_persists_across_calls(self, chroma_path, chroma_collection, chunk_docs, ingest_log_path):
        """dedup=True across two separate store_chunks calls skips duplicates."""
        store_chunks(chunk_docs, chroma_collection, dedup=True, ingest_log_path=ingest_log_path)
        second = store_chunks(chunk_docs, chroma_collection, dedup=True, ingest_log_path=ingest_log_path)
        assert second == 0


class TestRetrieve:
    def test_retrieve_with_query_vector(self, chroma_collection, chunk_docs, mock_embedder):
        store_chunks(chunk_docs, chroma_collection, dedup=False)

        qvec = mock_embedder.encode(["query about the document"])[0].tolist()

        results = retrieve("", chroma_collection, k=3, query_vector=qvec)
        assert len(results) > 0
        for r in results:
            assert "text" in r
            assert "metadata" in r
            assert "score" in r

    def test_retrieve_returns_top_k(self, chroma_collection, chunk_docs, mock_embedder):
        store_chunks(chunk_docs, chroma_collection, dedup=False)

        qvec = mock_embedder.encode(["query"])[0].tolist()
        results = retrieve("", chroma_collection, k=2, query_vector=qvec)
        assert len(results) == 2

    def test_retrieve_metadata_preserved(self, chroma_collection, chunk_docs, mock_embedder):
        store_chunks(chunk_docs, chroma_collection, dedup=False)

        qvec = mock_embedder.encode(["query"])[0].tolist()
        results = retrieve("", chroma_collection, k=1, query_vector=qvec)
        m = results[0]["metadata"]
        assert "source" in m
        assert "chunk_index" in m
        assert "chunk_hash" in m
