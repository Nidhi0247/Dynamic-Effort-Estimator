"""Text representation (S1): requirement text -> dense vector, reduced by a PCA fitted on the training window.

backend="minilm": frozen pretrained sentence encoder (all-MiniLM-L6-v2), embeddings cached on disk.
backend="tfidf":  TF-IDF + truncated SVD trained on the training window only (ablation / fallback).
"""
import hashlib
import os
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

CACHE = Path(__file__).resolve().parent.parent / "data" / "cache"
_MODEL = None


def _minilm():
    global _MODEL
    if _MODEL is None:
        from sentence_transformers import SentenceTransformer
        _MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    return _MODEL


def embed(texts, batch_size=128):
    """Embeddings with an on-disk cache keyed by text hash (texts are as-of snapshots, so the key is exact)."""
    CACHE.mkdir(parents=True, exist_ok=True)
    keys = [hashlib.sha1(t.encode("utf-8", "ignore")).hexdigest() for t in texts]
    store = CACHE / "minilm.npz"
    if store.exists():
        with np.load(store) as z:                 # close the handle (Windows cannot replace an open file)
            ck, cv = list(z["keys"]), np.array(z["vecs"])
    else:
        ck, cv = [], np.zeros((0, 384), np.float32)
    index = {k: i for i, k in enumerate(ck)}
    missing = sorted({k for k in keys if k not in index})
    if missing:
        lookup = {k: t for k, t in zip(keys, texts)}
        vecs = _minilm().encode([lookup[k] for k in missing], batch_size=batch_size, show_progress_bar=False,
                                normalize_embeddings=True).astype(np.float32)
        for k in missing:
            index[k] = len(index)
        cv = np.vstack([cv, vecs])
        ck = ck + missing
        tmp = store.with_name(f"minilm.{os.getpid()}.tmp.npz")
        np.savez(tmp, keys=np.array(ck), vecs=cv)
        os.replace(tmp, store)                    # atomic: parallel readers never see a half-written file
    return cv[[index[k] for k in keys]]


def minilm_available():
    try:
        import sentence_transformers  # noqa: F401
        return True
    except Exception:
        return False


class TextEncoder:
    def __init__(self, backend="minilm", dim=16, seed=0):
        self.backend = backend if (backend != "minilm" or minilm_available()) else "tfidf"
        self.dim, self.seed = dim, seed

    def fit(self, texts):
        texts = list(texts)
        if self.backend == "minilm":
            self.pca = PCA(self.dim, random_state=self.seed).fit(embed(texts))
        else:
            self.vec = TfidfVectorizer(max_features=20000, min_df=3, sublinear_tf=True, ngram_range=(1, 2))
            X = self.vec.fit_transform(texts)
            self.svd = TruncatedSVD(min(self.dim, X.shape[1] - 1), random_state=self.seed).fit(X)
        return self

    def transform(self, texts):
        texts = list(texts)
        if self.backend == "minilm":
            return self.pca.transform(embed(texts))
        return self.svd.transform(self.vec.transform(texts))
