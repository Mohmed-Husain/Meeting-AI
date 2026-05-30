from dataclasses import dataclass

import numpy as np
from sklearn.cluster import KMeans

from app.ai.clustering.centroid_selector import select_representative_indices
from app.ai.clustering.silhouette import select_best_k
from app.core.logger import get_logger

logger = get_logger("meeting_ai")


@dataclass
class CompressionResult:
    representative_indices: list[int]
    labels: list[int]
    best_k: int
    silhouette: float


class KMeansCompressor:
    def __init__(
        self,
        min_k: int = 2,
        max_k: int = 8,
        random_state: int = 42,
    ) -> None:
        self.min_k = max(2, min_k)
        self.max_k = max(self.min_k, max_k)
        self.random_state = random_state

    def compress(self, embeddings: list[list[float]]) -> CompressionResult | None:
        if not embeddings or len(embeddings) < self.min_k + 1:
            return None

        matrix = np.array(embeddings, dtype=np.float32)
        matrix = self._normalize(matrix)

        max_k = min(self.max_k, len(embeddings) - 1)
        if max_k < self.min_k:
            return None

        best_k, best_score = select_best_k(
            matrix=matrix,
            min_k=self.min_k,
            max_k=max_k,
            random_state=self.random_state,
        )
        if best_k is None:
            return None

        final_model = KMeans(
            n_clusters=best_k,
            random_state=self.random_state,
            n_init=10,
        )
        labels = final_model.fit_predict(matrix)
        representative_indices = select_representative_indices(
            matrix,
            labels,
            final_model.cluster_centers_,
        )
        logger.info(
            "KMeans compression selected k=%d (silhouette=%.3f)",
            best_k,
            best_score,
        )

        return CompressionResult(
            representative_indices=representative_indices,
            labels=labels.tolist(),
            best_k=best_k,
            silhouette=best_score,
        )

    def _normalize(self, matrix: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        return matrix / np.clip(norms, 1e-8, None)
