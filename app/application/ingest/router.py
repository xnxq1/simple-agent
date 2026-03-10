from fastapi import APIRouter

from app.logic.nodes.ingest.base import IngestState


class IngestRouter:
    def __init__(self, ingest_graph):
        self.ingest_graph = ingest_graph
        self.router = APIRouter(prefix="/ingest", tags=["ingest"])
        self.register_routes()

    def register_routes(self):
        self.router.post("/")(self.ingest_execute)

    async def ingest_execute(self, urls: list[str]):
        res = await self.ingest_graph.ainvoke(IngestState(
            urls=urls,
        ))
        return res['chunks']