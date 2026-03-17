import dataclasses
from datetime import datetime
from uuid import UUID


@dataclasses.dataclass
class ThreadSummary:
    id: UUID
    thread_id: str
    summary: str
    covered_episode_ids: list
    topics: list
    tools_used: list
    total_turns: int
    avg_context_score: float | None
    avg_faithfulness_score: float | None
    avg_answer_relevance_score: float | None
    created: datetime
