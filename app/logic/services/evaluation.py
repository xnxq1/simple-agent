"""Service for evaluating answer quality across multiple dimensions."""

import asyncio
import logging

import numpy as np
from langchain_core.embeddings import Embeddings
from langchain_core.messages import HumanMessage
from sentence_transformers import CrossEncoder

from app.domain.prompts import (
    AnswerRelevanceResult,
    ContextRelevance,
    ContextRelevanceResult,
    GroundnessResult,
    LLMAnswerRelevance,
    answer_relevance_system_prompt,
    answer_relevance_user_prompt,
    evaluate_groundness_system_prompt,
    evaluate_groundness_user_prompt,
)
from app.infra.llm.client import LLMClient

logger = logging.getLogger(__name__)


class EvaluationService:
    """Evaluates answer quality: context relevance, faithfulness, and answer relevance."""

    def __init__(
        self,
        llm_client: LLMClient,
        embed_model: Embeddings,
        cross_encoder_model: CrossEncoder,
    ):
        self.llm_client = llm_client
        self.embed_model = embed_model
        self.cross_encoder_model = cross_encoder_model

    async def evaluate(
        self, question: str, answer: str, retrieve_context: list[str]
    ) -> tuple[ContextRelevanceResult, GroundnessResult, AnswerRelevanceResult]:
        """Run all three evaluation methods in parallel.

        Args:
            question: Original user question
            answer: Generated answer
            retrieve_context: Retrieved context documents

        Returns:
            Tuple of (context_relevance_result, groundness_result, answer_relevance_result)
        """

        (
            context_relevance_result,
            groundness_result,
            answer_relevance_result,
        ) = await asyncio.gather(
            self.evaluate_context_relevance(question, retrieve_context),
            self.evaluate_faithfulness(context=retrieve_context, answer=answer),
            self.evaluate_answer_relevance(query=question, answer=answer),
        )
        return (context_relevance_result, groundness_result, answer_relevance_result)

    async def evaluate_context_relevance(
        self, query: str, docs: list[str]
    ) -> ContextRelevanceResult:
        """Score relevance of retrieved documents using CrossEncoder.

        Args:
            query: User question
            docs: Retrieved documents

        Returns:
            ContextRelevanceResult with per-doc scores and overall score
        """
        if not docs:
            return ContextRelevanceResult(context_relevance=[], context_score=0.0)

        # Prepare query-document pairs for CrossEncoder
        pairs = [[query, doc] for doc in docs]

        # Run CrossEncoder.predict in thread pool (blocking call)
        raw_scores = await asyncio.to_thread(
            self.cross_encoder_model.predict, sentences=pairs
        )

        # Normalize logits to [0, 1] using sigmoid
        scores = [float(1 / (1 + np.exp(-s))) for s in raw_scores]

        # Build per-document results
        result = [
            ContextRelevance(query=query, document=doc, score=score)
            for doc, score in zip(docs, scores)
        ]

        context_score = round(max(scores), 4) if scores else 0.0
        return ContextRelevanceResult(
            context_relevance=result, context_score=context_score
        )

    @staticmethod
    def cosine_similarity(a, b) -> float:
        """Compute cosine similarity between two vectors."""
        a = np.array(a)
        b = np.array(b)
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    async def evaluate_answer_relevance(
        self, query: str, answer: str
    ) -> AnswerRelevanceResult:
        """Score how well the answer addresses the original question.

        Args:
            query: Original user question
            answer: Generated answer

        Returns:
            AnswerRelevanceResult with generated questions and similarity score
        """
        try:
            result: LLMAnswerRelevance = await self.llm_client.completions_create(
                system_prompt=answer_relevance_system_prompt,
                messages=[
                    HumanMessage(
                        content=answer_relevance_user_prompt.format(answer=answer)
                    )
                ],
                response_class=LLMAnswerRelevance,
            )
            if not result.questions:
                return AnswerRelevanceResult(score=0.0, questions=[])
            embeddings = await self.embed_model.aembed_documents(
                [query] + result.questions
            )
            query_embed = embeddings[0]
            sim_questions_embeds = embeddings[1:]
            scores = []
            for embedding, llm_question in zip(sim_questions_embeds, result.questions):
                score = self.cosine_similarity(query_embed, embedding)
                scores.append(score)

            score = round(np.mean(scores), 2) if scores else 0.0
            return AnswerRelevanceResult(questions=result.questions, score=score)
        except Exception as e:
            logger.error(f"Answer relevance evaluation failed: {e}")
            return AnswerRelevanceResult(questions=[], score=0.0)

    async def evaluate_faithfulness(
        self, context: list[str], answer: str
    ) -> GroundnessResult:
        """Score how well the answer is grounded in retrieved context.

        Args:
            context: Retrieved context documents
            answer: Generated answer

        Returns:
            GroundnessResult with extracted claims and faithfulness score
        """
        try:
            user_prompt = evaluate_groundness_user_prompt.format(
                context=context, answer=answer
            )
            result: GroundnessResult = await self.llm_client.completions_create(
                system_prompt=evaluate_groundness_system_prompt,
                messages=[HumanMessage(content=user_prompt)],
                response_class=GroundnessResult,
            )
            return result
        except Exception as e:
            logger.error(f"Faithfulness evaluation failed: {e}")
            return GroundnessResult(claims=[], faithfulness_score=0.0)
