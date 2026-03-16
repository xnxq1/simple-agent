from uuid import UUID

from app.domain.threads import UserThread
from app.logic.handlers.base import BaseHandler
from app.logic.services.thread import ThreadService


class CreateThreadHandler(BaseHandler):
    def __init__(self, thread_service: ThreadService) -> None:
        self.thread_service = thread_service

    async def execute(self, user_id: UUID) -> UserThread:
        return await self.thread_service.create_thread(user_id=user_id)


class GetThreadHistoryHandler(BaseHandler):
    def __init__(self, thread_service: ThreadService) -> None:
        self.thread_service = thread_service

    async def execute(self, thread_id: str, user_id: UUID) -> list[dict]:
        return await self.thread_service.get_history(thread_id=thread_id, user_id=user_id)
