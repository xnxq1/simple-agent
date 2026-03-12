from uuid import UUID

from fastapi import APIRouter
from pydantic import BaseModel

from app.logic.handlers.topic import (
    CreateTopicHandler,
    GetTopicHandler,
    UpdateTopicHandler,
)


class UpdateTopic(BaseModel):
    archived: bool


class TopicRouter:
    def __init__(
        self,
        create_topic_handler: CreateTopicHandler,
        get_topic_handler: GetTopicHandler,
        update_topic_handler: UpdateTopicHandler,
    ) -> None:
        self.create_topic_handler = create_topic_handler
        self.get_topic_handler = get_topic_handler
        self.update_topic_handler = update_topic_handler
        self.router = APIRouter(prefix="/topics", tags=["topics"])
        self.register_routes()

    def register_routes(self):
        self.router.post("/")(self.create_topic)
        self.router.get("/")(self.get_topics)
        self.router.patch("/{entity_id}")(self.update_topic)

    async def create_topic(self, name: str):
        res = await self.create_topic_handler.execute(name=name)
        return res

    async def get_topics(self):
        res = await self.get_topic_handler.execute()
        return res

    async def update_topic(self, entity_id: UUID, payload: UpdateTopic):
        res = await self.update_topic_handler.execute(
            entity_id=entity_id, **payload.model_dump()
        )
        return res
