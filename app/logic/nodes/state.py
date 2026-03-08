import operator
from typing import Annotated

from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages
from pydantic import BaseModel, Field


class MessagesState(BaseModel):
    messages: Annotated[list, add_messages]
    llm_calls: int = 0

class IngestState(BaseModel):
    docs: list | None = Field(default_factory=list)
    urls: list[str]