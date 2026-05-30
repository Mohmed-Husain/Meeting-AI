import os

from app.core.logger import get_logger

logger = get_logger("meeting_ai")


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
