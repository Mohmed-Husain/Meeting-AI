import time
import uuid

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool

from app.ai.schemas.meeting_schema import (
    MeetingNotes,
    QueryRequest,
    QueryResponse,
    RetrievalChunk,
)
from app.ai.summarization.summarizer import get_summarizer
from app.core.logger import get_logger
from app.utils.file_loader import extract_text_from_path, save_upload

router = APIRouter(prefix="/api")
logger = get_logger("meeting_ai")


@router.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@router.post("/summarize", response_model=MeetingNotes)
async def summarize_transcript(file: UploadFile = File(...)) -> MeetingNotes:
    total_start = time.perf_counter()
    logger.info("Starting file loading...")
    try:
        load_start = time.perf_counter()
        path = await save_upload(file)
        load_duration = time.perf_counter() - load_start
        size_bytes = path.stat().st_size if path.exists() else 0
        logger.info("File loading completed in %.2fs (size=%s bytes)", load_duration, size_bytes)
    except ValueError as exc:
        logger.exception("File loading failed.")
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("File loading failed.")
        raise HTTPException(status_code=500, detail="Failed to save upload") from exc

    try:
        extract_start = time.perf_counter()
        transcript = extract_text_from_path(path)
        extract_duration = time.perf_counter() - extract_start
        logger.info("Transcript extraction completed in %.2fs (chars=%s)", extract_duration, len(transcript))
    except ValueError as exc:
        logger.exception("Transcript extraction failed.")
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Transcript extraction failed.")
        raise HTTPException(status_code=500, detail="Failed to read transcript") from exc

    if not transcript.strip():
        logger.error("Transcript is empty after extraction.")
        raise HTTPException(status_code=400, detail="Transcript is empty")

    summarizer = get_summarizer()
    meeting_id = f"meeting_{uuid.uuid4().hex[:8]}"
    original_filename = file.filename or "uploaded_transcript"
    metadata = {
        "source_filename": original_filename,
    }

    try:
        notes = await run_in_threadpool(summarizer.summarize, transcript, meeting_id, metadata)
    except Exception as exc:
        logger.exception("Summarization pipeline failed.")
        raise HTTPException(status_code=500, detail="Summarization failed") from exc

    logger.info("Total request completed in %.2fs", time.perf_counter() - total_start)
    return notes


@router.post("/query", response_model=QueryResponse)
async def query_meetings(request: QueryRequest) -> QueryResponse:
    logger.info("Received query request: '%s'", request.query)
    try:
        from app.ai.rag.query_chain import get_rag_answer_service
        from app.ai.rag.retriever import get_retriever_service

        retriever = get_retriever_service()
        answer_service = get_rag_answer_service()

        retrieved_chunks = await run_in_threadpool(
            lambda: retriever.retrieve(
                query=request.query,
                n_results=request.n_results,
                meeting_id=request.meeting_id,
            )
        )

        answer = await run_in_threadpool(
            lambda: answer_service.answer(
                question=request.query,
                retrieved_chunks=retrieved_chunks,
            )
        )

        source_chunks = []
        for chunk in retrieved_chunks:
            meta = chunk.get("metadata", {})
            source_chunks.append(
                RetrievalChunk(
                    chunk_id=chunk.get("id", ""),
                    text=chunk.get("document", ""),
                    meeting_id=meta.get("meeting_id", ""),
                    chunk_index=meta.get("chunk_index", 0),
                    date=meta.get("date", ""),
                    source_filename=meta.get("source_filename", ""),
                    distance=chunk.get("distance", 0.0),
                )
            )

        return QueryResponse(answer=answer, source_chunks=source_chunks)

    except Exception as exc:
        logger.exception("RAG query failed.")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/meetings")
async def list_meetings() -> list[dict]:
    logger.info("Received list meetings request")
    try:
        from app.ai.rag.vector_store import get_vector_store_service
        vector_store = get_vector_store_service()

        meetings = await run_in_threadpool(vector_store.get_all_meetings)
        return meetings
    except Exception as exc:
        logger.exception("Failed to list meetings.")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.delete("/meetings/{meeting_id}")
async def delete_meeting(meeting_id: str) -> dict:
    logger.info("Received delete meeting request for: %s", meeting_id)
    try:
        from app.ai.rag.vector_store import get_vector_store_service
        vector_store = get_vector_store_service()

        await run_in_threadpool(vector_store.delete_meeting_chunks, meeting_id)
        return {"status": "success", "message": f"Successfully deleted meeting {meeting_id}"}
    except Exception as exc:
        logger.exception("Failed to delete meeting.")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

