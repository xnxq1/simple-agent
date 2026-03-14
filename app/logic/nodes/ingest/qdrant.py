from app.logic.nodes.ingest.base import BaseIngestNode, IngestState
from app.logic.services.vector_store import VectorStoreService


class QdrantIngestNode(BaseIngestNode):
    def __init__(self, service: VectorStoreService):
        self.service = service

    async def execute(self, state: IngestState) -> dict:
        await self.service.upsert_chunks(state.chunks, state.embeddings)
        return {}
