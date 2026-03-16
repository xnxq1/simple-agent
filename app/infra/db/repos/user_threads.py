from sqlalchemy.ext.asyncio import AsyncEngine

from app.domain.threads import UserThread
from app.infra.db.models.user_threads import user_threads
from app.infra.db.repos.base import EntityRepo


class ThreadNotOwnedError(Exception):
    pass


class UserThreadsRepo(EntityRepo):
    db_entity = user_threads
    domain_entity = UserThread

    def __init__(self, engine: AsyncEngine) -> None:
        super().__init__(engine)
