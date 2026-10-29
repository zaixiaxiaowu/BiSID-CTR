"""Residual-quantized KMeans (RQ-KMeans) used to obtain Semantic IDs."""
from __future__ import annotations

import numpy as np
from sklearn.cluster import MiniBatchKMeans


class RQKMeans:
    def __init__(self, n_levels=3, n_clusters=256, random_state=2024, batch_size=4096, n_init=3, max_iter=100):
        self.n_levels = n_levels
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.batch_size = batch_size
        self.n_init = n_init
        self.max_iter = max_iter
        self.codebooks = []

    def fit(self, embeddings: np.ndarray) -> np.ndarray:
        residual = np.asarray(embeddings, dtype=np.float32)
        codes = np.zeros((residual.shape[0], self.n_levels), dtype=np.int64)
        self.codebooks = []
        for level in range(self.n_levels):
            n_clusters = min(self.n_clusters, max(2, residual.shape[0]))
            km = MiniBatchKMeans(
                n_clusters=n_clusters,
                random_state=self.random_state + level,
                batch_size=min(self.batch_size, residual.shape[0]),
                n_init=self.n_init,
                max_iter=self.max_iter,
            )
            pred = km.fit_predict(residual)
            centers = km.cluster_centers_.astype(np.float32)
            self.codebooks.append(centers)
            codes[:, level] = pred
            residual = residual - centers[pred]
        return codes

    def encode(self, embeddings: np.ndarray) -> np.ndarray:
        residual = np.asarray(embeddings, dtype=np.float32)
        codes = np.zeros((residual.shape[0], self.n_levels), dtype=np.int64)
        for level, centers in enumerate(self.codebooks):
            dist = ((residual[:, None, :] - centers[None, :, :]) ** 2).sum(-1)
            pred = dist.argmin(axis=1)
            codes[:, level] = pred
            residual = residual - centers[pred]
        return codes
