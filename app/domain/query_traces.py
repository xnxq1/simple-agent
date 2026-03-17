import dataclasses
from datetime import datetime
from uuid import UUID


@dataclasses.dataclass
class QueryTrace:
    id: UUID
    thread_id: str
    message_id: str
    question: str
    answer: str
    tools_used: list
    topics: list
    context_score: float | None
    faithfulness_score: float | None
    answer_relevance_score: float | None
    summary_id: UUID | None
    created: datetime
