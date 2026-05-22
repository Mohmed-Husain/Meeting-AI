import os
import re
import time
from functools import lru_cache

from app.ai.chains import build_map_chain, build_reduce_chain
from app.ai.chunker import TranscriptChunker
from app.ai.schemas import MeetingNotes
from app.core.logger import get_logger

FILLER_WORDS = [
    "uh",
    "um",
    "erm",
    "like",
    "you know",
    "i mean",
    "sort of",
    "kind of",
]

FILLER_PATTERN = re.compile(
    r"\b(?:" + "|".join(re.escape(word) for word in FILLER_WORDS) + r")\b",
    flags=re.IGNORECASE,
)

NOISE_PATTERN = re.compile(
    r"\[(?:inaudible|laughter|laughs|crosstalk|noise|applause|music)[^\]]*\]"
    r"|\((?:inaudible|laughter|laughs|crosstalk|noise|applause|music)[^)]*\)",
    flags=re.IGNORECASE,
)

SPEAKER_PATTERN = re.compile(r"^\s*([A-Za-z][A-Za-z0-9 _\-.]{0,40}):\s*(.*)$")

logger = get_logger("meeting_ai")


def _clean_text(text: str) -> str:
    cleaned = NOISE_PATTERN.sub(" ", text)
    cleaned = FILLER_PATTERN.sub("", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    return cleaned.strip()


def clean_transcript(transcript: str) -> str:
    if not transcript:
        return ""

    lines: list[str] = []
    last_blank = False

    for raw_line in transcript.splitlines():
        if not raw_line.strip():
            if not last_blank:
                lines.append("")
                last_blank = True
            continue

        last_blank = False
        speaker_match = SPEAKER_PATTERN.match(raw_line)

        if speaker_match:
            speaker = speaker_match.group(1).strip()
            content = _clean_text(speaker_match.group(2))
            if content:
                lines.append(f"{speaker}: {content}")
            else:
                lines.append(f"{speaker}:")
            continue

        cleaned_line = _clean_text(raw_line)
        if cleaned_line:
            lines.append(cleaned_line)

    return "\n".join(lines).strip()


def get_llm():
    provider = os.getenv("LLM_PROVIDER", "openai").strip().lower()
    logger.info("Using provider: %s", provider)

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        model = os.getenv("OLLAMA_MODEL", "llama3")
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        logger.info("Using model: %s", model)

        return ChatOllama(
            model=model,
            base_url=base_url,
            temperature=0.2,
        )

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        logger.info("Using model: %s", model)

        return ChatOpenAI(
            model=model,
            temperature=0.2,
        )

    logger.error("Unsupported provider: %s", provider)
    raise ValueError(f"Unsupported provider: {provider}")

    


class MeetingSummarizer:
    def __init__(self) -> None:
        chunk_size = int(os.getenv("CHUNK_SIZE", "3000"))
        chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "300"))
        self.max_concurrency = int(os.getenv("MAP_CONCURRENCY", "4"))
        self.inference_warn_seconds = float(os.getenv("INFERENCE_WARN_SECONDS", "15"))
        self.provider = os.getenv("LLM_PROVIDER", "openai").strip().lower()

        self.chunker = TranscriptChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        self.llm = get_llm()
        self.map_chain = build_map_chain(self.llm)
        self.reduce_chain = build_reduce_chain(self.llm)

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

        logger.info("Starting transcript chunking...")
        chunk_start = time.perf_counter()
        chunks = self.chunker.split_transcript(cleaned)
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
        try:
            import uuid
            from datetime import datetime
            from app.ai.embeddings import get_embedding_service
            from app.ai.vector_store import get_vector_store_service

            actual_meeting_id = meeting_id or f"meeting_{uuid.uuid4().hex[:8]}"
            actual_metadata = metadata or {}
            
            if "date" not in actual_metadata:
                actual_metadata["date"] = datetime.now().strftime("%Y-%m-%d")
            if "source_filename" not in actual_metadata:
                actual_metadata["source_filename"] = "uploaded_transcript"

            logger.info("Generating embeddings for transcript chunks and storing in ChromaDB...")
            embed_start = time.perf_counter()
            embedding_service = get_embedding_service()
            vector_store_service = get_vector_store_service()

            embeddings = embedding_service.embed_texts(chunks)
            
            metadata_list = []
            for idx in range(len(chunks)):
                chunk_meta = actual_metadata.copy()
                chunk_meta["chunk_index"] = idx
                metadata_list.append(chunk_meta)

            vector_store_service.store_chunks(
                meeting_id=actual_meeting_id,
                chunks=chunks,
                embeddings=embeddings,
                metadata_list=metadata_list,
            )
            logger.info("Ingestion completed in %.2fs", time.perf_counter() - embed_start)

        except Exception as exc:
            logger.warning("Failed to store chunks in vector store: %s. Proceeding with summarization.", str(exc))

        ai_start = time.perf_counter()

        logger.info("Starting map summarization...")
        map_start = time.perf_counter()
        try:
            partial_summaries = self._run_map(chunks)
        except Exception:
            logger.exception("Map summarization failed.")
            raise
        logger.info("Map summarization completed in %.2fs", time.perf_counter() - map_start)
        combined = "\n\n---\n\n".join(summary for summary in partial_summaries if summary)

        logger.info("Starting reduce summarization...")
        reduce_start = time.perf_counter()
        try:
            result = self.reduce_chain.invoke({"partial_summaries": combined})
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
