import hashlib
from pathlib import Path
from datetime import datetime, timezone

from chunking import chunk_markdown


class TestChunkMarkdown:
    def test_returns_list_of_dicts(self, sample_markdown, sample_metadata):
        chunks = chunk_markdown(sample_markdown, sample_metadata)
        assert isinstance(chunks, list)
        assert len(chunks) > 0
        for c in chunks:
            assert "text" in c
            assert "metadata" in c

    def test_produces_multiple_chunks(self, sample_markdown, sample_metadata):
        chunks = chunk_markdown(sample_markdown, sample_metadata)
        assert len(chunks) >= 4

    def test_chunk_metadata_shape(self, sample_markdown, sample_metadata):
        chunks = chunk_markdown(sample_markdown, sample_metadata)
        for c in chunks:
            m = c["metadata"]
            assert "source" in m
            assert "title" in m
            assert "doc_type" in m
            assert "chunk_index" in m
            assert "chunk_hash" in m
            assert "timestamp" in m
            assert m["source"] == "test_doc.md"
            assert m["title"] == "Test Document"
            assert m["doc_type"] == ".md"

    def test_chunk_indices_sequential(self, sample_markdown, sample_metadata):
        chunks = chunk_markdown(sample_markdown, sample_metadata)
        indices = [c["metadata"]["chunk_index"] for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_deterministic_hashes(self, sample_markdown, sample_metadata):
        chunks1 = chunk_markdown(sample_markdown, sample_metadata)
        chunks2 = chunk_markdown(sample_markdown, sample_metadata)
        for c1, c2 in zip(chunks1, chunks2):
            assert c1["metadata"]["chunk_hash"] == c2["metadata"]["chunk_hash"]

    def test_single_chunk_for_short_text(self, sample_metadata):
        chunks = chunk_markdown("Short text.", sample_metadata)
        assert len(chunks) == 1
        assert "Short text." in chunks[0]["text"]

    def test_long_text_splits_across_chunks(self, sample_metadata):
        long_text = "word " * 2000
        chunks = chunk_markdown(long_text, sample_metadata)
        assert len(chunks) > 1

    def test_empty_text_returns_empty(self, sample_metadata):
        chunks = chunk_markdown("", sample_metadata)
        assert chunks == []

    def test_chunk_hash_is_sha256_prefix(self, sample_markdown, sample_metadata):
        chunks = chunk_markdown(sample_markdown, sample_metadata)
        for c in chunks:
            h = c["metadata"]["chunk_hash"]
            assert len(h) == 16
            expected = hashlib.sha256(c["text"].encode("utf-8")).hexdigest()[:16]
            assert h == expected
