from functools import lru_cache

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

from app.ai.providers.llm_provider import get_llm
from app.ai.summarization.prompts import RAG_HUMAN_PROMPT, RAG_SYSTEM_PROMPT
from app.core.logger import get_logger

logger = get_logger("meeting_ai")


def build_rag_chain(llm) -> Runnable:
    """
    Build the RAG chain connecting prompt template, LLM, and output parser.
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", RAG_SYSTEM_PROMPT),
            ("human", RAG_HUMAN_PROMPT),
        ]
    )
    return prompt | llm | StrOutputParser()


class RAGAnswerService:
    def __init__(self) -> None:
        self.llm = get_llm()
        self.rag_chain = build_rag_chain(self.llm)

    def answer(self, question: str, retrieved_chunks: list[dict]) -> str:
        """
        Synthesize an answer using retrieved chunks as context.
        """
        if not retrieved_chunks:
            return "No relevant meeting context was found to answer this question."

        logger.info("Synthesizing RAG answer for question: '%s'", question)

        formatted_chunks = []
        for idx, item in enumerate(retrieved_chunks, start=1):
            doc = item["document"]
            meta = item["metadata"]
            source = meta.get("source_filename", "unknown")
            date = meta.get("date", "unknown")
            chunk_idx = meta.get("chunk_index", idx)

            chunk_header = f"--- CHUNK {chunk_idx} (Source: {source}, Date: {date}) ---"
            formatted_chunks.append(f"{chunk_header}\n{doc}")

        context_str = "\n\n".join(formatted_chunks)

        try:
            answer_text = self.rag_chain.invoke(
                {
                    "context": context_str,
                    "question": question,
                }
            )
            return answer_text
        except Exception:
            logger.exception("Failed to run RAG generation chain")
            raise


@lru_cache(maxsize=1)
def get_rag_answer_service() -> RAGAnswerService:
    return RAGAnswerService()
