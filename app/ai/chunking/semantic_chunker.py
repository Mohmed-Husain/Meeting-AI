import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from app.ai.chunking.rough_chunker import TranscriptChunker
from app.core.logger import get_logger

if TYPE_CHECKING:
    from app.ai.embeddings.embedding_service import EmbeddingService

logger = get_logger("meeting_ai")


@dataclass
class SemanticChunkResult:
    chunks: list[str]
    embeddings: list[list[float]] | None


class SemanticChunker:
    def __init__(self, embedding_service: "EmbeddingService") -> None:
        self.embedding_service = embedding_service
        self.similarity_threshold = float(os.getenv("SEMANTIC_SIMILARITY_THRESHOLD", "0.78"))
        self.semantic_min_chars = int(os.getenv("SEMANTIC_MIN_CHUNK_CHARS", "600"))
        self.semantic_max_chars = int(
            os.getenv("SEMANTIC_MAX_CHUNK_CHARS", os.getenv("CHUNK_SIZE", "3000"))
        )
        rough_size = int(os.getenv("ROUGH_CHUNK_SIZE", "1200"))
        rough_overlap = int(os.getenv("ROUGH_CHUNK_OVERLAP", "120"))
        self.rough_chunker = TranscriptChunker(
            chunk_size=rough_size,
            chunk_overlap=rough_overlap,
        )

    def split_and_embed(self, transcript: str) -> SemanticChunkResult:
        if not transcript.strip():
            return SemanticChunkResult([], [])

        rough_chunks = self.rough_chunker.split_transcript(transcript)
        if not rough_chunks:
            return SemanticChunkResult([], [])

        rough_embeddings = self._safe_embed(rough_chunks)
        if rough_embeddings is None:
            logger.warning("Semantic chunking fallback: embeddings unavailable.")
            return SemanticChunkResult(rough_chunks, None)

        if len(rough_chunks) == 1:
            return SemanticChunkResult(rough_chunks, rough_embeddings)

        semantic_chunks, semantic_embeddings = self._merge_rough_chunks(
            rough_chunks, rough_embeddings
        )
        logger.info(
            "Semantic chunking produced %d chunks from %d rough chunks",
            len(semantic_chunks),
            len(rough_chunks),
        )
        return SemanticChunkResult(semantic_chunks, semantic_embeddings)

    def _safe_embed(self, texts: list[str]) -> list[list[float]] | None:
        try:
            return self.embedding_service.embed_texts(texts)
        except Exception:
            logger.exception("Embedding generation failed during semantic chunking")
            return None

    def _merge_rough_chunks(
        self,
        rough_chunks: list[str],
        rough_embeddings: list[list[float]],
    ) -> tuple[list[str], list[list[float]]]:
        semantic_chunks: list[str] = []
        semantic_embeddings: list[list[float]] = []

        current_parts: list[str] = []
        current_embeds: list[list[float]] = []
        current_chars = 0

        for idx, chunk in enumerate(rough_chunks):
            if not current_parts:
                current_parts.append(chunk)
                current_embeds.append(rough_embeddings[idx])
                current_chars = len(chunk)
                continue

            similarity = self._cosine_similarity(
                rough_embeddings[idx - 1], rough_embeddings[idx]
            )
            split_on_similarity = (
                similarity < self.similarity_threshold
                and current_chars >= self.semantic_min_chars
            )
            split_on_max = current_chars + len(chunk) > self.semantic_max_chars and current_chars > 0

            if split_on_similarity or split_on_max:
                self._finalize_current(
                    current_parts,
                    current_embeds,
                    semantic_chunks,
                    semantic_embeddings,
                )
                current_parts = []
                current_embeds = []
                current_chars = 0

            current_parts.append(chunk)
            current_embeds.append(rough_embeddings[idx])
            current_chars += len(chunk)

        if current_parts:
            self._finalize_current(
                current_parts,
                current_embeds,
                semantic_chunks,
                semantic_embeddings,
            )

        return semantic_chunks, semantic_embeddings

    def _finalize_current(
        self,
        parts: list[str],
        embeds: list[list[float]],
        semantic_chunks: list[str],
        semantic_embeddings: list[list[float]],
    ) -> None:
        text = "\n\n".join(parts).strip()
        if not text:
            return

        semantic_chunks.append(text)
        semantic_embeddings.append(self._mean_embedding(embeds))

    def _cosine_similarity(self, left: list[float], right: list[float]) -> float:
        left_vec = np.array(left, dtype=np.float32)
        right_vec = np.array(right, dtype=np.float32)
        denom = (np.linalg.norm(left_vec) * np.linalg.norm(right_vec)) + 1e-8
        return float(np.dot(left_vec, right_vec) / denom)

    def _mean_embedding(self, embeds: list[list[float]]) -> list[float]:
        if not embeds:
            return []
        matrix = np.array(embeds, dtype=np.float32)
        mean = matrix.mean(axis=0)
        return mean.tolist()
