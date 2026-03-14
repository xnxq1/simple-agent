from app.logic.nodes.ingest.base import BaseIngestNode, IngestState
from app.logic.services.metadata_filling import MetadataFillingService


class MetadataFillingNode(BaseIngestNode):
    def __init__(self, service: MetadataFillingService):
        self.service = service

    async def execute(self, state: IngestState) -> dict:
        updated_chunks = await self.service.fill_topics(
            state.chunks, state.semantic_chunks
        )
        return {"chunks": updated_chunks}
