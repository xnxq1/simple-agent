import dataclasses
from datetime import datetime
from uuid import UUID


@dataclasses.dataclass
class UserThread:
    id: UUID
    user_id: UUID
    thread_id: str
    created: datetime
