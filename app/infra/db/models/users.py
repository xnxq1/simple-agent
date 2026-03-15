from sqlalchemy import Column, String, Table

from app.infra.db.utils import get_base_fields, metadata

users = Table(
    "users",
    metadata,
    *get_base_fields(),
    Column("name", String, nullable=False),
)
