from sqlalchemy.ext.asyncio import AsyncEngine

from app.domain.users import User
from app.infra.db.models.users import users
from app.infra.db.repos.base import EntityRepo


class UsersRepo(EntityRepo):
    db_entity = users
    domain_entity = User

    def __init__(self, engine: AsyncEngine) -> None:
        super().__init__(engine)
