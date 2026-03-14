import operator
from dataclasses import dataclass
from typing import Annotated

from langgraph.graph import add_messages
from pydantic import BaseModel, Field

from app.domain.prompts import GroundnessResult, ContextRelevanceResult, AnswerRelevanceResult


class MessagesState(BaseModel):
    messages: Annotated[list, add_messages]
    llm_calls: int = 0
    question: str
    answer: str | None = None
    retrieve_context: Annotated[list[str], operator.add] = []
    context_relevance_result: ContextRelevanceResult | None = None
    groundness_result: GroundnessResult | None = None
    answer_relevance_result: AnswerRelevanceResult | None = None
