from sqlalchemy import TIMESTAMP, Column, ForeignKey, Table, Text, UUID, text

from app.infra.db.utils import metadata

user_threads = Table(
    "user_threads",
    metadata,
    Column("id", UUID, primary_key=True, server_default=text("uuid_generate_v4()")),
    Column("user_id", UUID, ForeignKey("users.id"), nullable=False),
    Column("thread_id", Text, nullable=False, unique=True),
    Column(
        "created",
        TIMESTAMP(timezone=True),
        server_default=text("(now() at time zone 'utc')"),
        nullable=False,
    ),
)
