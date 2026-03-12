from sqlalchemy import Boolean, Column, String, Table

from app.infra.db.utils import get_base_fields, metadata

topics = Table(
    "topics",
    metadata,
    *get_base_fields(),
    Column("name", String, nullable=False),
    Column("is_active", Boolean, nullable=False),
)
