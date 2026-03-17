from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncEngine

from app.domain.thread_summaries import ThreadSummary
from app.infra.db.models.thread_summaries import thread_summaries
from app.infra.db.repos.base import EntityRepo
from app.infra.db.repos.exceptions import handle_db_errors


class ThreadSummariesRepo(EntityRepo):
    db_entity = thread_summaries
    domain_entity = ThreadSummary

    def __init__(self, engine: AsyncEngine) -> None:
        super().__init__(engine)

    @handle_db_errors
    async def get_latest(self, thread_id: str) -> ThreadSummary | None:
        query = (
            select(thread_summaries)
            .where(thread_summaries.c.thread_id == thread_id)
            .order_by(thread_summaries.c.created.desc())
            .limit(1)
        )
        row = await self.fetchrow(query)
        return ThreadSummary(**row) if row else None
