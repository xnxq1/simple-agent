import operator
from typing import Annotated

from langgraph.graph import add_messages
from pydantic import BaseModel

from app.domain.prompts import (
    AnswerRelevanceResult,
    ContextRelevanceResult,
    GroundnessResult,
)


class MessagesState(BaseModel):
    old_messages: list | None = None
    new_messages: Annotated[list, add_messages]
    llm_calls: int = 0
    question: str
    answer: str | None = None
    retrieve_context: Annotated[list[str], operator.add] = []
    context_relevance_result: ContextRelevanceResult | None = None
    groundness_result: GroundnessResult | None = None
    answer_relevance_result: AnswerRelevanceResult | None = None
    summary: str | None = None
    summary_id: str | None = None
