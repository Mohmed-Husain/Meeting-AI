import os
import time
from functools import lru_cache

from app.ai.chunking.chunk_manager import ChunkManager
from app.ai.clustering.kmeans_cluster import KMeansCompressor
from app.ai.rag.vector_store import get_vector_store_service
from app.ai.schemas.meeting_schema import MeetingNotes
from app.ai.summarization.chains import (
    build_map_chain,
    build_reduce_chain,
    build_reduce_text_chain,
)
from app.ai.summarization.hierarchical_reduce import HierarchicalReducer
from app.ai.providers.llm_provider import get_llm
from app.core.logger import get_logger
from app.utils.transcript_cleaner import clean_transcript

logger = get_logger("meeting_ai")


class MeetingSummarizer:
    def __init__(self) -> None:
        self.max_concurrency = int(os.getenv("MAP_CONCURRENCY", "4"))
        self.inference_warn_seconds = float(os.getenv("INFERENCE_WARN_SECONDS", "15"))
        self.provider = os.getenv("LLM_PROVIDER", "openai").strip().lower()

        self.chunk_manager = ChunkManager()
        self.llm = get_llm()
        self.map_chain = build_map_chain(self.llm)
        self.reduce_chain = build_reduce_chain(self.llm)
        self.reduce_text_chain = build_reduce_text_chain(self.llm)

        self.large_transcript_threshold = int(os.getenv("LARGE_TRANSCRIPT_THRESHOLD", "40"))
        self.kmeans_min_k = int(os.getenv("KMEANS_MIN_K", "2"))
        self.kmeans_max_k = int(os.getenv("KMEANS_MAX_K", "8"))
        self.kmeans_compressor = KMeansCompressor(
            min_k=self.kmeans_min_k,
            max_k=self.kmeans_max_k,
        )

        group_size = int(os.getenv("HIERARCHICAL_GROUP_SIZE", "6"))
        self.hierarchical_reducer = HierarchicalReducer(
            reduce_text_chain=self.reduce_text_chain,
            reduce_chain=self.reduce_chain,
            group_size=group_size,
        )

    def _run_map(self, chunks: list[str]) -> list[str]:
        total = len(chunks)
        results: list[str] = []

        for idx, chunk in enumerate(chunks, start=1):
            logger.info(
                "Starting AI inference for chunk %s/%s (size=%s chars)",
                idx,
                total,
                len(chunk),
            )
            step_start = time.perf_counter()
            try:
                result = self.map_chain.invoke({"transcript_chunk": chunk})
            except Exception:
                logger.exception("AI inference failed for chunk %s/%s", idx, total)
                raise
            duration = time.perf_counter() - step_start
            logger.info("Completed chunk %s/%s in %.2fs", idx, total, duration)

            if duration >= self.inference_warn_seconds:
                provider_hint = " (ollama)" if self.provider == "ollama" else ""
                logger.error(
                    "Slow AI inference%s for chunk %s/%s: %.2fs (threshold %.2fs)",
                    provider_hint,
                    idx,
                    total,
                    duration,
                    self.inference_warn_seconds,
                )

            results.append(result)

        return results

    def summarize(self, transcript: str, meeting_id: str = None, metadata: dict = None) -> MeetingNotes:
        total_start = time.perf_counter()

        logger.info("Starting transcript cleaning...")
        clean_start = time.perf_counter()
        cleaned = clean_transcript(transcript)
        logger.info("Transcript cleaning completed in %.2fs", time.perf_counter() - clean_start)

        logger.info("Starting semantic chunking...")
        chunk_start = time.perf_counter()
        chunk_bundle = self.chunk_manager.split_and_embed(cleaned)
        chunks = chunk_bundle.chunks
        chunk_embeddings = chunk_bundle.embeddings
        chunk_duration = time.perf_counter() - chunk_start
        logger.info("Chunking completed in %.2fs", chunk_duration)
        logger.info("Total chunks: %s", len(chunks))

        if chunks:
            sizes = [len(chunk) for chunk in chunks]
            logger.info(
                "Chunk size (chars): min=%s max=%s avg=%.0f",
                min(sizes),
                max(sizes),
                sum(sizes) / len(sizes),
            )

        if not chunks:
            logger.info("No chunks to process. Returning empty notes.")
            return MeetingNotes()

        # RAG Embedding and Vector Storage Ingestion (runs exactly once per upload)
        if chunk_embeddings and len(chunk_embeddings) == len(chunks):
            try:
                import uuid
                from datetime import datetime

                actual_meeting_id = meeting_id or f"meeting_{uuid.uuid4().hex[:8]}"
                actual_metadata = metadata or {}

                if "date" not in actual_metadata:
                    actual_metadata["date"] = datetime.now().strftime("%Y-%m-%d")
                if "source_filename" not in actual_metadata:
                    actual_metadata["source_filename"] = "uploaded_transcript"

                logger.info("Storing embeddings for transcript chunks in ChromaDB...")
                embed_start = time.perf_counter()
                vector_store_service = get_vector_store_service()

                metadata_list = []
                for idx in range(len(chunks)):
                    chunk_meta = actual_metadata.copy()
                    chunk_meta["chunk_index"] = idx
                    metadata_list.append(chunk_meta)

                vector_store_service.store_chunks(
                    meeting_id=actual_meeting_id,
                    chunks=chunks,
                    embeddings=chunk_embeddings,
                    metadata_list=metadata_list,
                )
                logger.info("Ingestion completed in %.2fs", time.perf_counter() - embed_start)

            except Exception as exc:
                logger.warning(
                    "Failed to store chunks in vector store: %s. Proceeding with summarization.",
                    str(exc),
                )
        else:
            logger.info("Skipping vector store ingestion (embeddings unavailable).")

        ai_start = time.perf_counter()

        summary_chunks = chunks
        if chunk_embeddings and len(chunks) > self.large_transcript_threshold:
            compression = self.kmeans_compressor.compress(chunk_embeddings)
            if compression and compression.representative_indices:
                summary_chunks = [chunks[idx] for idx in compression.representative_indices]
                logger.info(
                    "KMeans compression reduced %d -> %d chunks (k=%d, silhouette=%.3f)",
                    len(chunks),
                    len(summary_chunks),
                    compression.best_k,
                    compression.silhouette,
                )
            else:
                logger.info("KMeans compression skipped or failed; using all chunks.")
        elif len(chunks) <= self.large_transcript_threshold:
            logger.info(
                "KMeans compression skipped (chunks=%d, threshold=%d)",
                len(chunks),
                self.large_transcript_threshold,
            )

        logger.info("Starting map summarization...")
        map_start = time.perf_counter()
        try:
            partial_summaries = self._run_map(summary_chunks)
        except Exception:
            logger.exception("Map summarization failed.")
            raise
        logger.info("Map summarization completed in %.2fs", time.perf_counter() - map_start)

        logger.info("Starting hierarchical reduce summarization...")
        reduce_start = time.perf_counter()
        try:
            result = self.hierarchical_reducer.reduce(partial_summaries)
        except Exception:
            logger.exception("Reduce summarization failed.")
            raise

        reduce_duration = time.perf_counter() - reduce_start
        logger.info("Reduce summarization completed in %.2fs", reduce_duration)
        if reduce_duration >= self.inference_warn_seconds:
            provider_hint = " (ollama)" if self.provider == "ollama" else ""
            logger.error(
                "Slow reduce inference%s: %.2fs (threshold %.2fs)",
                provider_hint,
                reduce_duration,
                self.inference_warn_seconds,
            )

        ai_duration = time.perf_counter() - ai_start
        logger.info("Total AI processing time: %.2fs", ai_duration)
        logger.info("Final response generated.")
        logger.info("Total summarization completed in %.2fs", time.perf_counter() - total_start)

        return result


@lru_cache(maxsize=1)
def get_summarizer() -> MeetingSummarizer:
    return MeetingSummarizer()
