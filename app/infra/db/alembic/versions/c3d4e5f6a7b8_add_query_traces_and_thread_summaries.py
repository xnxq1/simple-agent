"""add query_traces and thread_summaries

Revision ID: c3d4e5f6a7b8
Revises: b2c3d4e5f6a7
Create Date: 2026-03-17 00:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "c3d4e5f6a7b8"
down_revision: Union[str, Sequence[str], None] = "b2c3d4e5f6a7"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "query_traces",
        sa.Column(
            "id",
            sa.UUID(),
            server_default=sa.text("uuid_generate_v4()"),
            nullable=False,
        ),
        sa.Column("thread_id", sa.Text(), nullable=False),
        sa.Column("message_id", sa.Text(), nullable=False),
        sa.Column("question", sa.Text(), nullable=False),
        sa.Column("answer", sa.Text(), nullable=False),
        sa.Column(
            "tools_used",
            sa.ARRAY(sa.Text()),
            nullable=False,
            server_default="{}",
        ),
        sa.Column(
            "topics",
            sa.ARRAY(sa.Text()),
            nullable=False,
            server_default="{}",
        ),
        sa.Column("context_score", sa.Float(), nullable=True),
        sa.Column("faithfulness_score", sa.Float(), nullable=True),
        sa.Column("answer_relevance_score", sa.Float(), nullable=True),
        sa.Column("summary_id", sa.UUID(), nullable=True),
        sa.Column(
            "created",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.text("(now() at time zone 'utc')"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_query_traces_thread_id", "query_traces", ["thread_id"])

    op.create_table(
        "thread_summaries",
        sa.Column(
            "id",
            sa.UUID(),
            server_default=sa.text("uuid_generate_v4()"),
            nullable=False,
        ),
        sa.Column("thread_id", sa.Text(), nullable=False),
        sa.Column("summary", sa.Text(), nullable=False),
        sa.Column("covered_episode_ids", sa.ARRAY(sa.UUID()), nullable=False),
        sa.Column(
            "topics",
            sa.ARRAY(sa.Text()),
            nullable=False,
            server_default="{}",
        ),
        sa.Column(
            "tools_used",
            sa.ARRAY(sa.Text()),
            nullable=False,
            server_default="{}",
        ),
        sa.Column(
            "total_turns",
            sa.Integer(),
            nullable=False,
            server_default="0",
        ),
        sa.Column("avg_context_score", sa.Float(), nullable=True),
        sa.Column("avg_faithfulness_score", sa.Float(), nullable=True),
        sa.Column("avg_answer_relevance_score", sa.Float(), nullable=True),
        sa.Column(
            "created",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.text("(now() at time zone 'utc')"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_thread_summaries_thread_id", "thread_summaries", ["thread_id"])


def downgrade() -> None:
    op.drop_index("ix_thread_summaries_thread_id", table_name="thread_summaries")
    op.drop_table("thread_summaries")
    op.drop_index("ix_query_traces_thread_id", table_name="query_traces")
    op.drop_table("query_traces")
