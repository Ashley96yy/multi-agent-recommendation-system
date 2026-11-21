"""Offline lexical-vector baseline and optional semantic embedding adapter.

The hashing backend is deterministic and requires no network or pretrained weights.
It is NOT a semantic language model; use MiniLMEmbedder for semantic retrieval.
"""

import hashlib
import re
from collections.abc import Sequence

import numpy as np


def normalized_vectors(matrix: np.ndarray, expected_rows: int) -> np.ndarray:
    """Validate, then L2-normalize vectors; zero rows stay zero."""
    try:
        array = np.asarray(matrix, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError("embedding backend returned nonnumeric vectors") from exc
    if array.ndim != 2 or array.shape[0] != expected_rows or array.shape[1] < 1:
        raise ValueError("embedding backend returned an invalid matrix shape")
    if not np.isfinite(array).all():
        raise ValueError("embedding backend returned non-finite vectors")
    norm = np.linalg.norm(array, axis=1, keepdims=True)
    return array / np.where(norm > 0, norm, 1.0)


class HashingEmbedder:
    """Stable hashed bag-of-words vectors, for CI, offline demos, and a lexical baseline."""

    def __init__(self, dimensions: int = 512):
        if not isinstance(dimensions, int) or isinstance(dimensions, bool) or dimensions < 8:
            raise ValueError("dimensions must be an integer >= 8")
        self.dimensions = dimensions
        self.name = f"lexical-hashing-v1-d{dimensions}"

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        vectors = np.zeros((len(texts), self.dimensions), dtype=np.float32)
        for i, text in enumerate(texts):
            if not isinstance(text, str):
                raise TypeError("each document/query must be text")
            for token in re.findall(r"[a-z0-9]+", text.casefold()):
                digest = hashlib.sha256(token.encode("utf-8")).digest()
                column = int.from_bytes(digest[:8], "big") % self.dimensions
                vectors[i, column] += 1.0
        return normalized_vectors(vectors, len(texts))


class MiniLMEmbedder:
    """Optional real semantic embeddings from a pinned SentenceTransformer revision.

    Dependencies and model weights are not installed/downloaded by the default CI.
    A full revision SHA is required to make semantic experiments reproducible.
    """

    MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(self, revision: str, device: str = "cpu"):
        if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError("revision must be a full 40-character lowercase model commit SHA")
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError('Install optional embeddings support: pip install -e ".[semantic]"') from exc
        self.name = f"{self.MODEL_ID}@{revision}"
        self._model = SentenceTransformer(self.MODEL_ID, revision=revision, device=device)

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            raise ValueError("MiniLM requires at least one text")
        return normalized_vectors(
            self._model.encode(
                list(texts),
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            ),
            len(texts),
        )
