import dataclasses

from app.domain.topics import BaseEntity


@dataclasses.dataclass
class User(BaseEntity):
    name: str
