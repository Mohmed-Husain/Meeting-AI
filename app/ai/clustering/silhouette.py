import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from app.core.logger import get_logger

logger = get_logger("meeting_ai")


def select_best_k(
    matrix: np.ndarray,
    min_k: int,
    max_k: int,
    random_state: int = 42,
) -> tuple[int | None, float]:
    best_score = -1.0
    best_k: int | None = None

    for k in range(min_k, max_k + 1):
        try:
            model = KMeans(
                n_clusters=k,
                random_state=random_state,
                n_init=10,
            )
            labels = model.fit_predict(matrix)
            score = silhouette_score(matrix, labels, metric="cosine")
            if score > best_score:
                best_score = score
                best_k = k
        except Exception:
            logger.exception("Silhouette scoring failed for k=%s", k)
            continue

    return best_k, best_score
