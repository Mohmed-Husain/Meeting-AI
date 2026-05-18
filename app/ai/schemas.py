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
