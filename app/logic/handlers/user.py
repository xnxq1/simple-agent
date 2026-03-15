from app.infra.db.repos.users import UsersRepo
from app.logic.handlers.base import BaseHandler


class CreateUserHandler(BaseHandler):
    def __init__(self, users_repo: UsersRepo) -> None:
        self.users_repo = users_repo

    async def execute(self, name: str) -> None:
        return await self.users_repo.insert(payload={"name": name})


class GetUsersHandler(BaseHandler):
    def __init__(self, users_repo: UsersRepo) -> None:
        self.users_repo = users_repo

    async def execute(self) -> list:
        return await self.users_repo.search()
