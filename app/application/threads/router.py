from uuid import UUID

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

from app.infra.db.repos.user_threads import ThreadNotOwnedError
from app.logic.handlers.thread import CreateThreadHandler, GetThreadHistoryHandler


class CreateThreadRequest(BaseModel):
    user_id: UUID


class ThreadRouter:
    def __init__(
        self,
        create_thread_handler: CreateThreadHandler,
        get_thread_history_handler: GetThreadHistoryHandler,
    ) -> None:
        self.create_thread_handler = create_thread_handler
        self.get_thread_history_handler = get_thread_history_handler
        self.router = APIRouter(prefix="/threads", tags=["threads"])
        self.register_routes()

    def register_routes(self):
        self.router.post("/")(self.create_thread)
        self.router.get("/{thread_id}/history")(self.get_thread_history)

    async def create_thread(self, payload: CreateThreadRequest):
        return await self.create_thread_handler.execute(user_id=payload.user_id)

    async def get_thread_history(self, thread_id: str, user_id: UUID):
        try:
            return await self.get_thread_history_handler.execute(
                thread_id=thread_id, user_id=user_id
            )
        except ThreadNotOwnedError:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Thread not found",
            )
