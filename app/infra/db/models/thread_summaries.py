from sqlalchemy import ARRAY, TIMESTAMP, UUID, Column, Float, Integer, Table, Text

from app.infra.db.utils import create_uuid, metadata, now_at_utc

thread_summaries = Table(
    "thread_summaries",
    metadata,
    Column("id", UUID, primary_key=True, server_default=create_uuid),
    Column("thread_id", Text, nullable=False, index=True),
    Column("summary", Text, nullable=False),
    Column("covered_episode_ids", ARRAY(UUID), nullable=False),
    Column("topics", ARRAY(Text), nullable=False, server_default="{}"),
    Column("tools_used", ARRAY(Text), nullable=False, server_default="{}"),
    Column("total_turns", Integer, nullable=False, server_default="0"),
    Column("avg_context_score", Float, nullable=True),
    Column("avg_faithfulness_score", Float, nullable=True),
    Column("avg_answer_relevance_score", Float, nullable=True),
    Column("created", TIMESTAMP(timezone=True), server_default=now_at_utc, nullable=False),
)
