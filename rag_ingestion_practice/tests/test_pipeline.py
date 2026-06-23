from pathlib import Path

import pytest

from pipeline import run_pipeline
from store import get_or_create_collection, retrieve


class TestPipelineSmoke:
    def test_e2e_with_txt_file(self, tmp_dir, chroma_path, ingest_log_path):
        """End-to-end: create a file, run pipeline, verify retrieval."""
        src_file = tmp_dir / "e2e_test.txt"
        src_file.write_text(
            "# Test Document\n\n"
            "This is a test document for end-to-end pipeline testing.\n"
            "It has enough content to verify the full flow works correctly.",
            encoding="utf-8",
        )

        total = run_pipeline(
            sources=[str(src_file)],
            reindex=False,
            chroma_path=chroma_path,
            ingest_log_path=ingest_log_path,
        )
        assert total > 0

    def test_reindex_clears_and_reingests(self, tmp_dir, chroma_path, ingest_log_path):
        """Running with --reindex clears previous data."""
        src_file = tmp_dir / "reindex_test.txt"
        src_file.write_text("# Doc\nSingle paragraph.", encoding="utf-8")

        run_pipeline(
            sources=[str(src_file)],
            reindex=False,
            chroma_path=chroma_path,
            ingest_log_path=ingest_log_path,
        )

        run_pipeline(
            sources=[str(src_file)],
            reindex=True,
            chroma_path=chroma_path,
            ingest_log_path=ingest_log_path,
        )

        collection = get_or_create_collection(chroma_path=chroma_path, collection_name="rag_docs")
        assert collection.count() > 0

    def test_retrieve_after_pipeline(self, tmp_dir, chroma_path, ingest_log_path):
        """Ingested content can be retrieved."""
        src_file = tmp_dir / "retrieval_test.txt"
        src_file.write_text(
            "# Machine Learning\n\n"
            "Machine learning is a method of data analysis that automates analytical model building.\n"
            "It is a branch of artificial intelligence based on the idea that systems can learn from data.",
            encoding="utf-8",
        )

        run_pipeline(
            sources=[str(src_file)],
            reindex=True,
            chroma_path=chroma_path,
            ingest_log_path=ingest_log_path,
        )

        collection = get_or_create_collection(chroma_path=chroma_path, collection_name="rag_docs")

        from embed import embed_query
        qvec = embed_query("machine learning")
        results = retrieve("", collection, k=3, query_vector=qvec)

        assert len(results) > 0
        assert any("machine learning" in r["text"].lower() for r in results)
