import textwrap
import tempfile
import hashlib
from pathlib import Path
from datetime import datetime, timezone
from unittest.mock import MagicMock

import numpy as np
import pytest
import chromadb
from chromadb.config import Settings

from chunking import chunk_markdown


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        yield Path(d)


@pytest.fixture
def chroma_path(tmp_dir):
    p = tmp_dir / "chroma_test"
    p.mkdir()
    return str(p)


@pytest.fixture
def ingest_log_path(tmp_dir):
    return str(tmp_dir / "ingest_log.json")


@pytest.fixture
def chroma_collection(chroma_path):
    client = chromadb.PersistentClient(
        path=chroma_path,
        settings=Settings(anonymized_telemetry=False),
    )
    collection = client.get_or_create_collection(
        name="test_collection",
        metadata={"hnsw:space": "cosine"},
    )
    yield collection
    try:
        client.delete_collection("test_collection")
    except Exception:
        pass


@pytest.fixture
def sample_markdown():
    return textwrap.dedent("""\
        # Introduction
        This is the introduction section with some text about the topic.

        ## Methodology
        We used a novel approach to solve the problem.

        ## Results
        The results were significant and showed improvement.

        ### Detailed Analysis
        A deeper look into the numbers reveals interesting patterns.

        # Conclusion
        In conclusion, this is a great document.
    """)


@pytest.fixture
def sample_metadata():
    return {
        "source": "test_doc.md",
        "title": "Test Document",
        "doc_type": ".md",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@pytest.fixture
def chunk_docs(sample_markdown, sample_metadata):
    return chunk_markdown(sample_markdown, sample_metadata)


@pytest.fixture
def mock_embedder():
    """Returns a fake SentenceTransformer that maps text → deterministic 384-d vectors."""
    model = MagicMock()
    EMBED_DIM = 384

    def _text_hash_seed(text: str) -> int:
        return int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)

    def fake_encode(texts, **kwargs):
        if isinstance(texts, str):
            texts = [texts]
        n = len(texts)
        vecs = np.zeros((n, EMBED_DIM), dtype=np.float32)
        for i, t in enumerate(texts):
            rng = np.random.default_rng(_text_hash_seed(t))
            vecs[i] = rng.normal(0, 0.1, EMBED_DIM)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        vecs = np.divide(vecs, norms, where=norms > 0, out=vecs)
        if kwargs.get("normalize_embeddings"):
            return vecs
        return vecs

    model.encode = fake_encode
    return model


@pytest.fixture
def mock_converter():
    """Returns a fake MarkItDown object for extract tests."""
    converter = MagicMock()

    def fake_convert(source, **kwargs):
        result = MagicMock()
        result.markdown = "# Mocked Title\n\nThis is mocked markdown content."
        result.title = "Mocked Title"
        return result

    converter.convert = fake_convert
    return converter


@pytest.fixture
def empty_mock_converter():
    """Returns a fake MarkItDown that returns empty markdown (simulates empty content)."""
    converter = MagicMock()

    def fake_convert(source, **kwargs):
        result = MagicMock()
        result.markdown = ""
        result.title = None
        return result

    converter.convert = fake_convert
    return converter
