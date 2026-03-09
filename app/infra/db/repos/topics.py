from sqlalchemy.ext.asyncio import AsyncEngine

from app.domain.topics import Topic
from app.infra.db.models import topics
from app.infra.db.repos.base import EntityRepo


class TopicsRepo(EntityRepo):
    db_entity = topics
    domain_entity = Topic

    def __init__(self, engine: AsyncEngine) -> None:
        super().__init__(engine)