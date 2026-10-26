"""Shared fastembed ONNX wrapper with LRU caching and lazy resource management."""

from collections import OrderedDict
import logging
import os
import threading
from typing import Any, List, Optional, Union

from fastembed import TextEmbedding
import numpy as np

from config import get_settings
from core.resources import get_resource_manager

logger = logging.getLogger("anya.core.embeddings")

RESOURCE_NAME = "fastembed"
DEFAULT_MODEL_NAME = "BAAI/bge-small-en-v1.5"
DEFAULT_CACHE_SIZE = 2048
EMBEDDING_DIM = 384


def _load_fastembed() -> TextEmbedding:
    """Instantiate the fastembed TextEmbedding model."""
    settings = get_settings()
    model_name = getattr(
        settings, "EMBEDDING_MODEL", os.getenv("EMBEDDING_MODEL", DEFAULT_MODEL_NAME)
    )
    logger.info("Initializing fastembed model '%s'...", model_name)
    return TextEmbedding(model_name=model_name)


def _unload_fastembed(instance: Any) -> None:
    """Release fastembed model instance from memory."""
    logger.info("Unloading fastembed model from memory")
    del instance


def register_embedding_resource() -> None:
    """Register fastembed with the central ResourceManager if not already present."""
    settings = get_settings()
    timeout_s = getattr(settings, "LLM_IDLE_UNLOAD_S", 300)
    res_mgr = get_resource_manager()
    if RESOURCE_NAME not in res_mgr._resources:
        res_mgr.register(
            RESOURCE_NAME,
            loader=_load_fastembed,
            unloader=_unload_fastembed,
            timeout_s=float(timeout_s),
        )


class EmbeddingCache:
    """Thread-safe LRU cache mapping text strings to 1D float32 numpy embeddings."""

    def __init__(self, maxsize: int = DEFAULT_CACHE_SIZE) -> None:
        self.maxsize = maxsize
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, text: str) -> Optional[np.ndarray]:
        """Fetch embedding from cache, updating LRU order."""
        with self._lock:
            if text in self._cache:
                self._cache.move_to_end(text)
                return self._cache[text]
            return None

    def put(self, text: str, embedding: np.ndarray) -> None:
        """Store embedding in cache, evicting oldest item if capacity is exceeded."""
        with self._lock:
            self._cache[text] = np.asarray(embedding, dtype=np.float32)
            self._cache.move_to_end(text)
            if len(self._cache) > self.maxsize:
                self._cache.popitem(last=False)

    def clear(self) -> None:
        """Clear all entries from cache."""
        with self._lock:
            self._cache.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._cache)


_CACHE = EmbeddingCache(maxsize=DEFAULT_CACHE_SIZE)

# Ensure resource registration on module import
register_embedding_resource()


def clear_cache() -> None:
    """Clear the embedding LRU cache."""
    _CACHE.clear()


def get_cache_size() -> int:
    """Return the current number of cached embeddings."""
    return len(_CACHE)


def embed(texts: Union[List[str], str]) -> np.ndarray:
    """Compute dense float32 embeddings for a list of strings or single string with LRU caching.

    Args:
        texts: A string or list of strings to embed.

    Returns:
        A 2D numpy array of shape (N, 384) with float32 values.
    """
    if isinstance(texts, str):
        text_list = [texts]
    else:
        text_list = list(texts)

    if not text_list:
        return np.empty((0, EMBEDDING_DIM), dtype=np.float32)

    results: List[Optional[np.ndarray]] = [None] * len(text_list)
    missing_indices: List[int] = []
    missing_texts: List[str] = []

    # Check LRU cache first
    for i, text in enumerate(text_list):
        cached = _CACHE.get(text)
        if cached is not None:
            results[i] = cached
        else:
            missing_indices.append(i)
            missing_texts.append(text)

    # Compute missing embeddings via lazily loaded fastembed model
    if missing_texts:
        res_mgr = get_resource_manager()
        model: TextEmbedding = res_mgr.get(RESOURCE_NAME)
        computed_embeddings = list(model.embed(missing_texts))

        for idx, text, emb in zip(missing_indices, missing_texts, computed_embeddings):
            emb_arr = np.asarray(emb, dtype=np.float32)
            _CACHE.put(text, emb_arr)
            results[idx] = emb_arr

    # Assemble and return 2D array
    stacked = np.stack([res for res in results if res is not None], axis=0)
    return stacked.astype(np.float32)


def cosine_similarity(
    a: Union[np.ndarray, List[float]],
    b: Union[np.ndarray, List[float]],
) -> Union[float, np.ndarray]:
    """Calculate cosine similarity between two 1D vectors or a 1D vector and a 2D matrix.

    Args:
        a: 1D query vector.
        b: 1D target vector or 2D matrix of shape (M, dim).

    Returns:
        A float similarity score between [-1.0, 1.0] if both are 1D,
        or a 1D numpy array of similarity scores if b is 2D.
    """
    vec_a = np.asarray(a, dtype=np.float32)
    vec_b = np.asarray(b, dtype=np.float32)

    if vec_a.ndim == 1 and vec_b.ndim == 1:
        norm_a = float(np.linalg.norm(vec_a))
        norm_b = float(np.linalg.norm(vec_b))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))

    if vec_a.ndim == 1 and vec_b.ndim == 2:
        if len(vec_b) == 0:
            return np.empty((0,), dtype=np.float32)
        norm_a = float(np.linalg.norm(vec_a))
        norm_b = np.linalg.norm(vec_b, axis=1)
        denom = norm_a * norm_b
        denom = np.where(denom == 0, 1e-9, denom)
        sims = np.dot(vec_b, vec_a) / denom
        return sims.astype(np.float32)

    raise ValueError(f"Unsupported dimensions for cosine_similarity: a={vec_a.shape}, b={vec_b.shape}")
