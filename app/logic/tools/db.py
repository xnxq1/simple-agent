from app.domain.topics import Topic
from app.infra.db.repos.topics import TopicsRepo


class DBTools:
    def __init__(self, topics_repo: TopicsRepo):
        self.topics_repo = topics_repo

    async def get_available_topics(self) -> list[Topic]:
        return await self.topics_repo.search(archived=False, is_active=True)