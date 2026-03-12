import dataclasses
from datetime import datetime
from uuid import UUID


@dataclasses.dataclass
class BaseEntity:
    id: UUID
    created: datetime
    updated: datetime
    archived: bool


@dataclasses.dataclass
class Topic(BaseEntity):
    name: str
    is_active: bool
