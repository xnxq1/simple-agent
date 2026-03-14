from app.logic.nodes.ingest.base import BaseIngestNode, IngestState
from app.logic.services.embedding import EmbeddingService


class EmbeddingNode(BaseIngestNode):
    def __init__(self, service: EmbeddingService):
        self.service = service

    async def execute(self, state: IngestState) -> dict:
        embeddings = await self.service.embed_chunks(state.chunks)
        return {"embeddings": embeddings}
