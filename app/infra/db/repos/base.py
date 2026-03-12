import abc
from contextlib import asynccontextmanager
from datetime import datetime
from uuid import UUID

from sqlalchemy import column, select
from sqlalchemy.ext.asyncio import AsyncEngine

from app.infra.db.repos.exceptions import handle_db_errors


class BaseRepo(abc.ABC):
    def __init__(self, engine: AsyncEngine) -> None:
        self.engine = engine

    @asynccontextmanager
    async def transaction(self):
        async with self.engine.begin() as conn:
            yield conn

    async def fetch(self, query) -> list:
        async with self.transaction() as conn:
            result = await conn.execute(query)
            result = result.fetchall()
            return [dict(r._mapping) for r in result]

    async def fetchrow(self, query) -> dict | None:
        async with self.transaction() as conn:
            result = await conn.execute(query)
            result = result.fetchone()
            return dict(result._mapping) if result else None


class EntityRepo(BaseRepo):
    db_entity = None
    domain_entity = None

    def _get_filter_bool_expression(self, filter_name, filter_value, base_query):
        if filter_name in base_query.columns:
            return column(filter_name).__eq__(filter_value)

        split_by_underscore = filter_name.split("_")
        sign = split_by_underscore.pop()
        col_name = "_".join(split_by_underscore)
        col = column(col_name)

        if sign in {"lt", "le", "gt", "ge", "ne"}:
            return getattr(col, f"__{sign}__")(filter_value)
        elif sign == "in":
            return col.in_(filter_value)
        elif sign == "notin":
            return ~col.in_(filter_value)
        elif sign == "is":
            return col.is_(filter_value)
        elif sign == "isnot":
            return col.is_not(filter_value)
        elif sign == "like":
            return col.like(filter_value)
        elif sign == "ilike":
            return col.ilike(filter_value)

        raise ValueError(f"Unknown filter name ({filter_name})")

    def _apply_filters(self, query, **filters):
        for filter_name, filter_value in filters.items():
            query = query.where(
                self._get_filter_bool_expression(filter_name, filter_value, query)
            )

        return query

    @handle_db_errors
    async def search(self, **filters) -> list[domain_entity]:
        query = self._apply_filters(select(self.db_entity), **filters)
        res = await self.fetch(query)
        return [self.domain_entity(**r) for r in res]

    @handle_db_errors
    async def search_first_row(self, **filters) -> domain_entity:
        res = await self.search(**filters)
        return res[0] if res else None

    @handle_db_errors
    async def insert(self, payload: dict) -> domain_entity:
        query = self.db_entity.insert().values(payload).returning(self.db_entity)
        res = await self.fetchrow(query)
        return self.domain_entity(**res)

    @handle_db_errors
    async def update_by_id(self, entity_id: UUID, **payload) -> dict:
        update_query = (
            self.db_entity.update()
            .values(updated=datetime.utcnow(), **payload)
            .returning(self.db_entity)
            .where(self.db_entity.c.id == entity_id)
        )
        return await self.fetchrow(update_query)
