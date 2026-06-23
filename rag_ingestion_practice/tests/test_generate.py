from generate import build_rag_prompt


def test_build_rag_prompt_includes_query_context_and_sources():
    chunks = [
        {
            "text": "This document explains a simple RAG ingestion pipeline.",
            "metadata": {
                "title": "sample",
                "source": "sample.txt",
            },
        }
    ]

    prompt = build_rag_prompt("What is this document about?", chunks)

    assert "What is this document about?" in prompt
    assert "simple RAG ingestion pipeline" in prompt
    assert "sample.txt" in prompt
    assert "Use only the context" in prompt
