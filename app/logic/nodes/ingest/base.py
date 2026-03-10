import abc
import dataclasses
from typing import List
from uuid import UUID

from pydantic import BaseModel, Field


@dataclasses.dataclass
class Chunk:
    id: UUID
    text: str
    metadata: dict

@dataclasses.dataclass
class SemanticChunk:
    metadata: dict
    text: str
    chunk_ids: List[UUID]

class IngestState(BaseModel):
    documents: list | None = Field(default_factory=list)
    urls: list[str]
    chunks: list[Chunk] | None = Field(default_factory=list)
    semantic_chunks: list[SemanticChunk] | None = Field(default_factory=list)
    embeddings: list[list[float]] | None = Field(default_factory=list)

class BaseIngestNode(abc.ABC):

    @abc.abstractmethod
    async def execute(self, state: IngestState) -> dict:
        ...