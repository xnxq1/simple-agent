from sqlalchemy import ARRAY, TIMESTAMP, UUID, Column, Float, Table, Text

from app.infra.db.utils import create_uuid, metadata, now_at_utc

query_traces = Table(
    "query_traces",
    metadata,
    Column("id", UUID, primary_key=True, server_default=create_uuid),
    Column("thread_id", Text, nullable=False, index=True),
    Column("message_id", Text, nullable=False),
    Column("question", Text, nullable=False),
    Column("answer", Text, nullable=False),
    Column("tools_used", ARRAY(Text), nullable=False, server_default="{}"),
    Column("topics", ARRAY(Text), nullable=False, server_default="{}"),
    Column("context_score", Float, nullable=True),
    Column("faithfulness_score", Float, nullable=True),
    Column("answer_relevance_score", Float, nullable=True),
    Column("summary_id", UUID, nullable=True),
    Column("created", TIMESTAMP(timezone=True), server_default=now_at_utc, nullable=False),
)
