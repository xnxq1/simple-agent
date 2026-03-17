import uuid
from uuid import UUID

from app.domain.threads import UserThread
from app.infra.db.repos.query_traces import QueryTracesRepo
from app.infra.db.repos.user_threads import ThreadNotOwnedError, UserThreadsRepo


class ThreadService:
    def __init__(
        self, user_threads_repo: UserThreadsRepo, query_traces_repo: QueryTracesRepo
    ) -> None:
        self.user_threads_repo = user_threads_repo
        self.query_traces_repo = query_traces_repo

    async def create_thread(self, user_id: UUID) -> UserThread:
        return await self.user_threads_repo.insert(
            payload={"user_id": user_id, "thread_id": str(uuid.uuid4())}
        )

    async def get_history(self, thread_id: str, user_id: UUID) -> list[dict]:
        thread = await self.user_threads_repo.search_first_row(
            thread_id=thread_id, user_id=user_id
        )
        if thread is None:
            raise ThreadNotOwnedError(
                f"Thread {thread_id} not found for user {user_id}"
            )

        traces = await self.query_traces_repo.get_by_thread_id(thread_id)
        history = []
        for trace in traces:
            history.append({"role": "human", "content": trace.question})
            history.append({"role": "assistant", "content": trace.answer})
        return history
