from uuid import UUID

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncEngine

from app.infra.db.models.query_traces import query_traces
from app.infra.db.repos.base import BaseRepo
from app.infra.db.repos.exceptions import handle_db_errors


class QueryTracesRepo(BaseRepo):
    db_entity = query_traces

    def __init__(self, engine: AsyncEngine) -> None:
        super().__init__(engine)

    @handle_db_errors
    async def insert(
        self,
        thread_id: str,
        message_id: str,
        question: str,
        answer: str,
        tools_used: list[str],
        topics: list[str],
        context_score: float | None,
        faithfulness_score: float | None,
        answer_relevance_score: float | None,
    ) -> dict:
        query = (
            query_traces.insert()
            .values(
                thread_id=thread_id,
                message_id=message_id,
                question=question,
                answer=answer,
                tools_used=tools_used,
                topics=topics,
                context_score=context_score,
                faithfulness_score=faithfulness_score,
                answer_relevance_score=answer_relevance_score,
            )
            .returning(query_traces)
        )
        return await self.fetchrow(query)

    @handle_db_errors
    async def get_unsummarized(self, thread_id: str) -> list[dict]:
        query = select(query_traces).where(
            query_traces.c.thread_id == thread_id,
            query_traces.c.summary_id.is_(None),
        )
        return await self.fetch(query)

    @handle_db_errors
    async def get_by_summary_id(self, summary_id: UUID) -> list[dict]:
        query = select(query_traces).where(
            query_traces.c.summary_id == summary_id,
        )
        return await self.fetch(query)

    @handle_db_errors
    async def set_summary_id(self, trace_ids: list[UUID], summary_id: UUID) -> None:
        query = (
            update(query_traces)
            .where(query_traces.c.id.in_(trace_ids))
            .values(summary_id=summary_id)
        )
        async with self.transaction() as conn:
            await conn.execute(query)
