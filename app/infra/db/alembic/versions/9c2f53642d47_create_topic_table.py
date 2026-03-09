"""create topic table

Revision ID: 9c2f53642d47
Revises: 
Create Date: 2026-03-09 15:18:07.057838

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '9c2f53642d47'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.execute('CREATE EXTENSION IF NOT EXISTS "uuid-ossp"')
    op.create_table('topics',
    sa.Column('id', sa.UUID(), server_default=sa.text('uuid_generate_v4()'), nullable=False),
    sa.Column('created', sa.TIMESTAMP(timezone=True), server_default=sa.text("(now() at time zone 'utc')"), nullable=False),
    sa.Column('updated', sa.TIMESTAMP(timezone=True), server_default=sa.text("(now() at time zone 'utc')"), nullable=False),
    sa.Column('archived', sa.Boolean(), server_default='false', nullable=False),
    sa.Column('name', sa.String(), nullable=False),
    sa.Column('is_active', sa.Boolean(), nullable=False),
    sa.PrimaryKeyConstraint('id')
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table('topics')
    op.execute('DROP EXTENSION IF EXISTS "uuid-ossp"')
