from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncEngine

from app.infra.db.models.thread_summaries import thread_summaries
from app.infra.db.repos.base import BaseRepo
from app.infra.db.repos.exceptions import handle_db_errors


class ThreadSummariesRepo(BaseRepo):
    db_entity = thread_summaries

    def __init__(self, engine: AsyncEngine) -> None:
        super().__init__(engine)

    @handle_db_errors
    async def get_latest(self, thread_id: str) -> dict | None:
        query = (
            select(thread_summaries)
            .where(thread_summaries.c.thread_id == thread_id)
            .order_by(thread_summaries.c.created.desc())
            .limit(1)
        )
        return await self.fetchrow(query)

    @handle_db_errors
    async def insert(
        self,
        thread_id: str,
        summary: str,
        covered_episode_ids: list[UUID],
        topics: list[str],
        tools_used: list[str],
        total_turns: int,
        avg_context_score: float | None,
        avg_faithfulness_score: float | None,
        avg_answer_relevance_score: float | None,
    ) -> dict:
        query = (
            thread_summaries.insert()
            .values(
                thread_id=thread_id,
                summary=summary,
                covered_episode_ids=covered_episode_ids,
                topics=topics,
                tools_used=tools_used,
                total_turns=total_turns,
                avg_context_score=avg_context_score,
                avg_faithfulness_score=avg_faithfulness_score,
                avg_answer_relevance_score=avg_answer_relevance_score,
            )
            .returning(thread_summaries)
        )
        return await self.fetchrow(query)
