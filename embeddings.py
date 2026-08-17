"""
Sentence-embedding generation with a persistent on-disk cache.

Every text we embed is hashed (sha256). If we've embedded that exact text
before (in any previous run), we reuse the stored vector instead of calling
the model again. This means re-running the pipeline on the same complaint
data is fast after the first run - only genuinely new/changed VOC text
gets embedded.
"""
import hashlib
import pickle

import numpy as np

from config import CACHE_DIR, EMBEDDING_MODEL_NAME


def _sanitize_model_name(name: str) -> str:
    """
    Model names like 'BAAI/bge-base-en-v1.5' contain a '/', which - if used
    directly in a path - gets interpreted as a subdirectory that doesn't
    exist ('embedding_cache/BAAI/bge-base-en-v1.5.pkl'), causing a
    FileNotFoundError on save. Flatten it into a single safe filename instead.
    """
    return name.replace("/", "__").replace("\\", "__")


_CACHE_FILE = CACHE_DIR / f"{_sanitize_model_name(EMBEDDING_MODEL_NAME)}.pkl"


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _load_cache() -> dict:
    if _CACHE_FILE.exists():
        with open(_CACHE_FILE, "rb") as f:
            return pickle.load(f)
    return {}


def _save_cache(cache: dict) -> None:
    tmp_path = _CACHE_FILE.with_suffix(".tmp")
    with open(tmp_path, "wb") as f:
        pickle.dump(cache, f)
    tmp_path.replace(_CACHE_FILE)


class EmbeddingModel:
    """Lazy-loads the sentence-transformers model only when actually needed."""

    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        self.model_name = model_name
        self._model = None
        # e5 models were trained expecting every input prefixed with
        # "query: " or "passage: " - without it you lose a real chunk of
        # the accuracy gain that makes e5 worth using. We're encoding one
        # kind of text (VOC complaints) purely for classification, not
        # query-vs-document retrieval, so "query: " uniformly is correct
        # here. bge models don't need this for plain classification use.
        self._prefix = "query: " if "e5" in model_name.lower() else ""

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def encode(self, texts: list[str], batch_size: int = 64, show_progress: bool = True) -> np.ndarray:
        """
        Return an (n, dim) array of embeddings for `texts`, using the disk
        cache wherever possible and only running the model on cache misses.
        """
        texts = [self._prefix + t if isinstance(t, str) else self._prefix for t in texts]
        cache = _load_cache()

        hashes = [_hash_text(t) for t in texts]
        missing_idx = [i for i, h in enumerate(hashes) if h not in cache]

        if missing_idx:
            model = self._get_model()
            missing_texts = [texts[i] for i in missing_idx]
            new_vectors = model.encode(
                missing_texts,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True,
            )
            for i, vec in zip(missing_idx, new_vectors):
                cache[hashes[i]] = vec
            _save_cache(cache)

        return np.vstack([cache[h] for h in hashes])