from app.ai.clustering.kmeans_cluster import CompressionResult, KMeansCompressor


class ClusterManager:
    def __init__(
        self,
        min_k: int = 2,
        max_k: int = 8,
        random_state: int = 42,
    ) -> None:
        self.compressor = KMeansCompressor(
            min_k=min_k,
            max_k=max_k,
            random_state=random_state,
        )

    def compress_embeddings(self, embeddings: list[list[float]]) -> CompressionResult | None:
        return self.compressor.compress(embeddings)
