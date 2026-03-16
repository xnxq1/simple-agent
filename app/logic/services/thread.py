import uuid
from uuid import UUID

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from app.domain.threads import UserThread
from app.infra.db.repos.user_threads import ThreadNotOwnedError, UserThreadsRepo


class ThreadService:
    def __init__(self, user_threads_repo: UserThreadsRepo, checkpointer: AsyncPostgresSaver) -> None:
        self.user_threads_repo = user_threads_repo
        self.checkpointer = checkpointer

    async def create_thread(self, user_id: UUID) -> UserThread:
        return await self.user_threads_repo.insert(
            payload={"user_id": user_id, "thread_id": str(uuid.uuid4())}
        )

    async def get_history(self, thread_id: str, user_id: UUID) -> list[dict]:
        thread = await self.user_threads_repo.search_first_row(
            thread_id=thread_id, user_id=user_id
        )
        if thread is None:
            raise ThreadNotOwnedError(f"Thread {thread_id} not found for user {user_id}")

        config = {"configurable": {"thread_id": thread_id}}
        checkpoint_tuple = await self.checkpointer.aget_tuple(config)
        if checkpoint_tuple is None:
            return []

        messages = checkpoint_tuple.checkpoint.get("channel_values", {}).get("messages", [])
        history = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                history.append({"role": "human", "content": msg.content})
            elif isinstance(msg, AIMessage) and msg.content:
                history.append({"role": "assistant", "content": msg.content})
        return history
