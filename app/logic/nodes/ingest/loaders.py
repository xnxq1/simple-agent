from app.logic.nodes.ingest.base import BaseIngestNode, IngestState
from app.logic.services.web_loader import WebLoaderService


class WebLoaderNode(BaseIngestNode):
    def __init__(self, service: WebLoaderService):
        self.service = service

    async def execute(self, state: IngestState) -> dict:
        documents = await self.service.load_urls(state.urls)
        return {"documents": documents}
