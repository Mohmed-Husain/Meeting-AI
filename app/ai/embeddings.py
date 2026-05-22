import os
from functools import lru_cache
from langchain_openai import OpenAIEmbeddings
from app.core.logger import get_logger

logger = get_logger("meeting_ai")

class EmbeddingService:
    def __init__(self) -> None:
        model_name = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        api_key = os.getenv("OPENAI_API_KEY")
        
        if not api_key:
            logger.error("OPENAI_API_KEY environment variable is not set!")
            raise ValueError("OPENAI_API_KEY is required for embedding generation.")
            
        logger.info("Initializing OpenAIEmbeddings with model: %s", model_name)
        self.embeddings = OpenAIEmbeddings(
            model=model_name,
            openai_api_key=api_key,
        )

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """
        Generate vector embeddings for a list of transcript chunks.
        """
        if not texts:
            return []
        
        logger.info("Generating embeddings for %d text chunks", len(texts))
        return self.embeddings.embed_documents(texts)

    def embed_query(self, query: str) -> list[float]:
        """
        Generate vector embedding for a single user query.
        """
        logger.info("Generating embedding for query")
        return self.embeddings.embed_query(query)

@lru_cache(maxsize=1)
def get_embedding_service() -> EmbeddingService:
    return EmbeddingService()
