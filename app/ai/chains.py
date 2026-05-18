from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

from app.ai.prompts import (
    MAP_HUMAN_PROMPT,
    MAP_SYSTEM_PROMPT,
    REDUCE_HUMAN_PROMPT,
    REDUCE_SYSTEM_PROMPT,
)
from app.ai.schemas import MeetingNotes


def build_map_chain(llm) -> Runnable:
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", MAP_SYSTEM_PROMPT),
            ("human", MAP_HUMAN_PROMPT),
        ]
    )
    return prompt | llm | StrOutputParser()


def build_reduce_chain(llm) -> Runnable:
    parser = PydanticOutputParser(pydantic_object=MeetingNotes)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", REDUCE_SYSTEM_PROMPT),
            ("human", REDUCE_HUMAN_PROMPT),
        ]
    ).partial(format_instructions=parser.get_format_instructions())

    return prompt | llm | parser
