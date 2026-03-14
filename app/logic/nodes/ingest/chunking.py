from app.logic.nodes.ingest.base import BaseIngestNode, IngestState
from app.logic.services.chunking import ChunkingService


class SemanticChunkingNode(BaseIngestNode):
    def __init__(self, service: ChunkingService):
        self.service = service

    async def execute(self, state: IngestState) -> dict:
        chunks, semantic_chunks = await self.service.chunk_documents(state.documents)
        return {"chunks": chunks, "semantic_chunks": semantic_chunks}
