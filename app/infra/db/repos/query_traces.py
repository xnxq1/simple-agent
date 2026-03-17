from uuid import UUID

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncEngine
from sqlalchemy import asc

from app.domain.query_traces import QueryTrace
from app.infra.db.models.query_traces import query_traces
from app.infra.db.repos.base import EntityRepo
from app.infra.db.repos.exceptions import handle_db_errors


class QueryTracesRepo(EntityRepo):
    db_entity = query_traces
    domain_entity = QueryTrace

    def __init__(self, engine: AsyncEngine) -> None:
        super().__init__(engine)

    @handle_db_errors
    async def get_unsummarized(self, thread_id: str) -> list[QueryTrace]:
        query = select(query_traces).where(
            query_traces.c.thread_id == thread_id,
            query_traces.c.summary_id.is_(None),
        )
        rows = await self.fetch(query)
        return [QueryTrace(**r) for r in rows]

    @handle_db_errors
    async def get_by_summary_id(self, summary_id: UUID) -> list[QueryTrace]:
        query = select(query_traces).where(
            query_traces.c.summary_id == summary_id,
        )
        rows = await self.fetch(query)
        return [QueryTrace(**r) for r in rows]

    @handle_db_errors
    async def get_by_thread_id(self, thread_id: str) -> list[QueryTrace]:
        query = (
            select(query_traces)
            .where(query_traces.c.thread_id == thread_id)
            .order_by(asc(query_traces.c.created))
        )
        rows = await self.fetch(query)
        return [QueryTrace(**r) for r in rows]

    @handle_db_errors
    async def set_summary_id(self, trace_ids: list[UUID], summary_id: UUID) -> None:
        query = (
            update(query_traces)
            .where(query_traces.c.id.in_(trace_ids))
            .values(summary_id=summary_id)
        )
        async with self.transaction() as conn:
            await conn.execute(query)
