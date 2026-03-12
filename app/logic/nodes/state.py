from typing import Annotated

from langgraph.graph import add_messages
from pydantic import BaseModel, Field


class MessagesState(BaseModel):
    messages: Annotated[list, add_messages]
    llm_calls: int = 0
    question: str
    answer: str | None = None
    retrieve_context: list[str] = Field(default_factory=list)
