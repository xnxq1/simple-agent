from sqlalchemy import TIMESTAMP, UUID, Boolean, Column, MetaData, text


now_at_utc = text("(now() at time zone 'utc')")
create_uuid = text("uuid_generate_v4()")
metadata = MetaData()


def get_base_fields() -> tuple:
    return (
        Column("id", UUID, primary_key=True, server_default=create_uuid),
        Column("created", TIMESTAMP(timezone=True), server_default=now_at_utc, nullable=False),
        Column("updated", TIMESTAMP(timezone=True), server_default=now_at_utc, nullable=False),
        Column("archived", Boolean, server_default="false", nullable=False),
    )
