import os
from dataclasses import dataclass

from app.ai.chunking.rough_chunker import TranscriptChunker
from app.ai.chunking.semantic_chunker import SemanticChunker
from app.ai.embeddings.embedding_service import get_embedding_service
from app.core.logger import get_logger

logger = get_logger("meeting_ai")


@dataclass
class ChunkBundle:
    chunks: list[str]
    embeddings: list[list[float]] | None


class ChunkManager:
    def __init__(self) -> None:
        chunk_size = int(os.getenv("CHUNK_SIZE", "3000"))
        chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "300"))
        self.fallback_chunker = TranscriptChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def split_and_embed(self, transcript: str) -> ChunkBundle:
        if not transcript.strip():
            return ChunkBundle([], None)

        try:
            embedding_service = get_embedding_service()
            semantic_chunker = SemanticChunker(embedding_service=embedding_service)
            result = semantic_chunker.split_and_embed(transcript)
            if result.embeddings is not None:
                return ChunkBundle(result.chunks, result.embeddings)
            logger.info("Semantic chunking returned no embeddings. Falling back to rough chunking.")
        except Exception as exc:
            logger.warning(
                "Semantic chunking failed: %s. Falling back to rough chunking.",
                str(exc),
            )

        chunks = self.fallback_chunker.split_transcript(transcript)
        return ChunkBundle(chunks, None)
