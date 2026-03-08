import dataclasses
import operator
from typing import Annotated

from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages
from pydantic import BaseModel, Field


class MessagesState(BaseModel):
    messages: Annotated[list, add_messages]
    llm_calls: int = 0

@dataclasses.dataclass
class Chunk:
    text: str
    metadata: dict

class IngestState(BaseModel):
    documents: list | None = Field(default_factory=list)
    urls: list[str]
    chunks: list[Chunk] | None = Field(default_factory=list)
    embeddings: list[list[float]] | None = Field(default_factory=list)