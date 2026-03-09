from app.infra.db.repos.topics import TopicsRepo
from app.logic.handlers.base import BaseHandler


class CreateTopicHandler(BaseHandler):

    def __init__(self, topic_repo: TopicsRepo) -> None:
        self.topic_repo = topic_repo

    async def execute(self, name: str) -> None:
        return await self.topic_repo.insert(payload={'name': name, 'is_active': True})

