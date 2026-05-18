import time

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool

from app.ai.schemas import MeetingNotes
from app.ai.summarizer import get_summarizer
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
    try:
        notes = await run_in_threadpool(summarizer.summarize, transcript)
    except Exception as exc:
        logger.exception("Summarization pipeline failed.")
        raise HTTPException(status_code=500, detail="Summarization failed") from exc

    logger.info("Total request completed in %.2fs", time.perf_counter() - total_start)
    return notes
