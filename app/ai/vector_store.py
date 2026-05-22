import os
from functools import lru_cache
import chromadb
from app.core.logger import get_logger

logger = get_logger("meeting_ai")

class VectorStoreService:
    def __init__(self) -> None:
        db_path = os.getenv("CHROMA_DB_PATH", "chroma_db")
        logger.info("Initializing ChromaDB persistent client at: %s", db_path)
        
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection_name = "meeting_chunks"
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name
        )

    def store_chunks(
        self,
        meeting_id: str,
        chunks: list[str],
        embeddings: list[list[float]],
        metadata_list: list[dict],
    ) -> None:
        """
        Store transcript chunks, embeddings, and metadata into ChromaDB.
        Uses idempotent chunk IDs to avoid duplication.
        """
        if not chunks or not embeddings:
            logger.warning("No chunks or embeddings provided to store_chunks")
            return

        if len(chunks) != len(embeddings) or len(chunks) != len(metadata_list):
            raise ValueError("Size mismatch between chunks, embeddings, and metadata list")

        logger.info(
            "Upserting %d chunks for meeting %s into ChromaDB",
            len(chunks),
            meeting_id,
        )

        ids = [f"{meeting_id}_chunk_{idx}" for idx in range(len(chunks))]

        # Ensure every metadata dict contains meeting_id
        for metadata in metadata_list:
            metadata["meeting_id"] = meeting_id

        try:
            self.collection.upsert(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadata_list,
                documents=chunks,
            )
            logger.info("Successfully stored %d chunks in vector store", len(chunks))
        except Exception:
            logger.exception("Failed to store chunks in ChromaDB")
            raise

    def similarity_search(
        self,
        query_embedding: list[float],
        n_results: int = 5,
        meeting_id: str = None,
    ) -> list[dict]:
        """
        Perform a similarity search in ChromaDB using a query embedding.
        Optionally filter by meeting_id.
        """
        logger.info(
            "Performing similarity search in ChromaDB (n_results=%d, meeting_id=%s)",
            n_results,
            meeting_id,
        )

        where_clause = {}
        if meeting_id:
            where_clause["meeting_id"] = meeting_id

        try:
            # collection.query returns a dict with 'documents', 'metadatas', 'distances', 'ids'
            # Each is a list of lists since we can query multiple embeddings. We only pass one query_embedding.
            query_args = {
                "query_embeddings": [query_embedding],
                "n_results": n_results,
            }
            if where_clause:
                query_args["where"] = where_clause

            results = self.collection.query(**query_args)

            formatted_results = []
            if results and results.get("documents"):
                documents = results["documents"][0]
                metadatas = results["metadatas"][0] if results.get("metadatas") else [{}] * len(documents)
                distances = results["distances"][0] if results.get("distances") else [0.0] * len(documents)
                ids = results["ids"][0] if results.get("ids") else [""] * len(documents)

                for doc, meta, dist, cid in zip(documents, metadatas, distances, ids):
                    formatted_results.append({
                        "id": cid,
                        "document": doc,
                        "metadata": meta,
                        "distance": dist,
                    })

            logger.info("Found %d results in similarity search", len(formatted_results))
            return formatted_results

        except Exception:
            logger.exception("Similarity search in ChromaDB failed")
            raise

    def delete_meeting_chunks(self, meeting_id: str) -> None:
        """
        Delete all stored chunks for a specific meeting.
        """
        logger.info("Deleting chunks for meeting %s from ChromaDB", meeting_id)
        try:
            self.collection.delete(where={"meeting_id": meeting_id})
            logger.info("Successfully deleted chunks for meeting %s", meeting_id)
        except Exception:
            logger.exception("Failed to delete chunks for meeting %s", meeting_id)
            raise

    def get_all_meetings(self) -> list[dict]:
        """
        Get all unique meetings stored in ChromaDB by scanning metadata.
        Returns a list of dicts with meeting_id and metadata info.
        """
        logger.info("Fetching all stored meetings from ChromaDB")
        try:
            # We can get all metadatas in the collection
            results = self.collection.get(include=["metadatas"])
            metadatas = results.get("metadatas", [])
            
            meetings_map = {}
            for meta in metadatas:
                if not meta or "meeting_id" not in meta:
                    continue
                mid = meta["meeting_id"]
                if mid not in meetings_map:
                    meetings_map[mid] = {
                        "meeting_id": mid,
                        "date": meta.get("date", ""),
                        "source_filename": meta.get("source_filename", ""),
                    }
            
            return list(meetings_map.values())
        except Exception:
            logger.exception("Failed to get stored meetings from ChromaDB")
            raise

@lru_cache(maxsize=1)
def get_vector_store_service() -> VectorStoreService:
    return VectorStoreService()
