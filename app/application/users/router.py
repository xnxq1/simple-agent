from fastapi import APIRouter

from app.logic.handlers.user import CreateUserHandler, GetUsersHandler


class UserRouter:
    def __init__(
        self,
        create_user_handler: CreateUserHandler,
        get_users_handler: GetUsersHandler,
    ) -> None:
        self.create_user_handler = create_user_handler
        self.get_users_handler = get_users_handler
        self.router = APIRouter(prefix="/users", tags=["users"])
        self.register_routes()

    def register_routes(self):
        self.router.post("/")(self.create_user)
        self.router.get("/")(self.get_users)

    async def create_user(self, name: str):
        return await self.create_user_handler.execute(name=name)

    async def get_users(self):
        return await self.get_users_handler.execute()
