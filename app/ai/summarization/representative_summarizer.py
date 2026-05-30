from app.core.logger import get_logger

logger = get_logger("meeting_ai")


class RepresentativeSummarizer:
    def __init__(self, map_chain) -> None:
        self.map_chain = map_chain

    def summarize(self, chunk: str) -> str:
        if not chunk.strip():
            return ""

        logger.info("Summarizing representative chunk (size=%s chars)", len(chunk))
        return self.map_chain.invoke({"transcript_chunk": chunk})
