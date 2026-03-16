import uuid
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncEngine

from app.infra.db.models.user_threads import user_threads
from app.infra.db.repos.base import BaseRepo

class UserThreadsRepo(BaseRepo):
    def __init__(self, engine: AsyncEngine) -> None:
        super().__init__(engine)
