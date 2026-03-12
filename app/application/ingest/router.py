from fastapi import APIRouter
from pydantic import BaseModel

from app.logic.nodes.ingest.base import IngestState


class IngestRequest(BaseModel):
    urls: list[str]


class IngestRouter:
    def __init__(self, ingest_graph):
        self.ingest_graph = ingest_graph
        self.router = APIRouter(prefix="/ingest", tags=["ingest"])
        self.register_routes()

    def register_routes(self):
        self.router.post("/")(self.ingest_execute)

    async def ingest_execute(self, request: IngestRequest):
        res = await self.ingest_graph.ainvoke(
            IngestState(
                urls=request.urls,
            )
        )
        return res["chunks"]
