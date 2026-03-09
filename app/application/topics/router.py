from fastapi import APIRouter

from app.logic.handlers.topic import CreateTopicHandler
from app.logic.nodes.state import IngestState


class TopicRouter:
    def __init__(self, create_topic_handler: CreateTopicHandler) -> None:
        self.create_topic_handler = create_topic_handler
        self.router = APIRouter(prefix="/topics", tags=["topics"])
        self.register_routes()

    def register_routes(self):
        self.router.post("/")(self.create_topic)

    async def create_topic(self, name: str):
        res = await self.create_topic_handler.execute(name=name)
        return res