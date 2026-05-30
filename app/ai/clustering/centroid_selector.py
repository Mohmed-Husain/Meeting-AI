import numpy as np


def select_representative_indices(
    matrix: np.ndarray,
    labels: np.ndarray,
    centers: np.ndarray,
) -> list[int]:
    representative_indices: list[int] = []

    for cluster_id in range(centers.shape[0]):
        cluster_indices = np.where(labels == cluster_id)[0]
        if cluster_indices.size == 0:
            continue

        cluster_vectors = matrix[cluster_indices]
        center = centers[cluster_id]
        distances = np.linalg.norm(cluster_vectors - center, axis=1)
        best_index = int(cluster_indices[int(distances.argmin())])
        representative_indices.append(best_index)

    return sorted(set(representative_indices))
