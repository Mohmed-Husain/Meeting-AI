from pydantic import BaseModel, Field


class AgendaItem(BaseModel):
    title: str = Field(default="", description="Agenda title with highlighted keywords.")
    description: str = Field(default="", description="One-line agenda description.")


class ActionItem(BaseModel):
    task: str = Field(default="", description="Action item task.")
    assigned_to: str = Field(default="", description="Assignee name if available.")
    deadline: str = Field(default="", description="Deadline if mentioned.")


class MeetingNotes(BaseModel):
    agendas: list[AgendaItem] = Field(default_factory=list)
    summary: list[str] = Field(default_factory=list)
    action_plan: list[ActionItem] = Field(default_factory=list)


class QueryRequest(BaseModel):
    query: str = Field(..., description="The user query or question about meetings.")
    meeting_id: str | None = Field(default=None, description="Optional meeting ID to restrict the search context.")
    n_results: int = Field(default=5, description="Number of context chunks to retrieve.")


class RetrievalChunk(BaseModel):
    chunk_id: str = Field(default="", description="The unique ID of the chunk.")
    text: str = Field(..., description="The raw transcript chunk text.")
    meeting_id: str = Field(..., description="The meeting ID this chunk belongs to.")
    chunk_index: int = Field(default=0, description="The sequential index of this chunk in the transcript.")
    date: str = Field(default="", description="The date of the meeting.")
    source_filename: str = Field(default="", description="The source file name.")
    distance: float = Field(default=0.0, description="The vector distance or similarity score.")


class QueryResponse(BaseModel):
    answer: str = Field(..., description="The context-aware AI generated answer.")
    source_chunks: list[RetrievalChunk] = Field(default_factory=list, description="The list of semantically retrieved context chunks.")
