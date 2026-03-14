from app.domain.prompts import (
    AnswerRelevanceResult,
    ContextRelevanceResult,
    GroundnessResult,
)
from app.logic.nodes.base import BaseLLMNode
from app.logic.nodes.state import MessagesState
from app.logic.services.evaluation import EvaluationService


class Evaluator(BaseLLMNode):
    def __init__(self, service: EvaluationService):
        self.service = service

    async def execute(self, state: MessagesState) -> dict:
        if not state.retrieve_context:
            return {
                "context_relevance_result": ContextRelevanceResult(
                    context_relevance=[], context_score=0.0
                ),
                "groundness_result": GroundnessResult(
                    claims=[], faithfulness_score=0.0
                ),
                "answer_relevance_result": AnswerRelevanceResult(
                    questions=[], score=0.0
                ),
            }
        (
            context_relevance_result,
            groundness_result,
            answer_relevance_result,
        ) = await self.service.evaluate(
            state.question, state.answer, state.retrieve_context
        )
        return {
            "context_relevance_result": context_relevance_result,
            "groundness_result": groundness_result,
            "answer_relevance_result": answer_relevance_result,
        }
