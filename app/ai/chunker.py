from langchain_text_splitters import RecursiveCharacterTextSplitter

class TranscriptChunker:
    def __init__(
        self,
        chunk_size: int = 3000,
        chunk_overlap: int = 300,
    ):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=[
                "\n\n",
                "\n",
                ". ",
                " ",
                ""
            ]
        )

    def split_transcript(self, transcript: str) -> list[str]:
        """
        Split large transcript into smaller optimized chunks.
        """

        if not transcript.strip():
            return []

        chunks = self.splitter.split_text(transcript)

        return [chunk for chunk in chunks if chunk.strip()]