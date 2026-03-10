from uuid import UUID

from app.infra.db.repos.topics import TopicsRepo
from app.logic.handlers.base import BaseHandler


class CreateTopicHandler(BaseHandler):

    def __init__(self, topic_repo: TopicsRepo) -> None:
        self.topic_repo = topic_repo

    async def execute(self, name: str) -> None:
        return await self.topic_repo.insert(payload={'name': name, 'is_active': True})


class GetTopicHandler(BaseHandler):
    def __init__(self, topic_repo: TopicsRepo) -> None:
        self.topic_repo = topic_repo

    async def execute(self) -> list:
        return await self.topic_repo.search()

class UpdateTopicHandler(BaseHandler):
    def __init__(self, topic_repo: TopicsRepo) -> None:
        self.topic_repo = topic_repo

    async def execute(self, entity_id: UUID, **payload) -> dict:
        return await self.topic_repo.update_by_id(entity_id=entity_id, **payload)