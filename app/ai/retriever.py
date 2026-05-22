from functools import lru_cache

from app.ai.embeddings import get_embedding_service
from app.ai.vector_store import get_vector_store_service
from app.core.logger import get_logger

logger = get_logger("meeting_ai")

class RetrieverService:
    def __init__(self) -> None:
        self.embedding_service = get_embedding_service()
        self.vector_store_service = get_vector_store_service()

    def retrieve(
        self,
        query: str,
        n_results: int = 5,
        meeting_id: str = None,
    ) -> list[dict]:
        """
        Retrieve semantically relevant transcript chunks for a query.
        Ties together embedding generation and similarity search.
        """
        if not query.strip():
            logger.warning("Empty query passed to retrieve")
            return []

        logger.info(
            "Retrieving context for query: '%s' (n_results=%d, meeting_id=%s)",
            query,
            n_results,
            meeting_id,
        )

        try:
            # Step 1: Generate embedding for the query
            query_embedding = self.embedding_service.embed_query(query)
            
            # Step 2: Search the vector store
            results = self.vector_store_service.similarity_search(
                query_embedding=query_embedding,
                n_results=n_results,
                meeting_id=meeting_id,
            )
            
            return results
        except Exception:
            logger.exception("Retrieval failed for query: %s", query)
            raise

@lru_cache(maxsize=1)
def get_retriever_service() -> RetrieverService:
    return RetrieverService()

