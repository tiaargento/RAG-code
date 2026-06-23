import numpy as np
import pytest

from embed import embed_chunks, embed_query


class TestEmbedMocked:
    def test_embed_chunks_shape(self, mock_embedder):
        texts = ["hello world", "foo bar baz"]
        vecs = embed_chunks(texts, model=mock_embedder)
        assert isinstance(vecs, np.ndarray)
        assert vecs.shape == (2, 384)

    def test_embed_chunks_single(self, mock_embedder):
        vecs = embed_chunks(["single text"], model=mock_embedder)
        assert vecs.shape == (1, 384)

    def test_embed_query_returns_list(self, mock_embedder):
        vec = embed_query("test query", model=mock_embedder)
        assert isinstance(vec, list)
        assert len(vec) == 384
        assert all(isinstance(v, float) for v in vec)

    def test_embed_query_normalized(self, mock_embedder):
        vec = embed_query("test query", model=mock_embedder)
        norm = np.linalg.norm(vec)
        assert abs(norm - 1.0) < 1e-5

    def test_same_text_same_vector(self, mock_embedder):
        """Deterministic mock: identical text → identical vector."""
        vecs = embed_chunks(["same text", "same text"], model=mock_embedder)
        np.testing.assert_array_equal(vecs[0], vecs[1])


@pytest.mark.integration
class TestEmbedReal:
    """One real integration test — loads the actual model once."""

    def test_real_embed_dimension(self):
        texts = ["real test sentence"]
        vecs = embed_chunks(texts)
        assert vecs.shape == (1, 384)
        norm = np.linalg.norm(vecs[0])
        assert abs(norm - 1.0) < 1e-4
