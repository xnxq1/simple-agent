import abc
import asyncio
import dataclasses

from langchain_core.embeddings import Embeddings
from langchain_core.messages import HumanMessage

from app.domain.prompts import GroundnessResult, evaluate_groundness_user_prompt, evaluate_groundness_system_prompt, \
    ContextRelevanceResult, ContextRelevance, answer_relevance_system_prompt, answer_relevance_user_prompt, \
    LLMAnswerRelevance, AnswerRelevanceResult
from app.infra.llm.client import LLMClient
from app.logic.nodes.base import BaseLLMNode
from app.logic.nodes.state import MessagesState
import numpy as np

class Evaluator(BaseLLMNode):
    def __init__(self, llm_client: LLMClient, embed_model: Embeddings):
        self.llm_client = llm_client
        self.embed_model = embed_model
        self.relevance_threshold = 0.6

    async def execute(self, state: MessagesState) -> dict:
        if not state.retrieve_context:
            return {
                'context_relevance_result': ContextRelevanceResult(context_relevance=[], context_score=0.0),
                'groundness_result': GroundnessResult(claims=[], faithfulness_score=0.0),
                'answer_relevance_result': AnswerRelevanceResult(questions=[], score=0.0),
            }
        context_relevance_result, groundness_result, answer_relevance_result = await asyncio.gather(
            self.evaluate_context_relevance(state.question, state.retrieve_context),
            self.evaluate_faithfulness(context=state.retrieve_context, answer=state.answer),
            self.evaluate_answer_relevance(query=state.question, answer=state.answer),
        )
        return {
            'context_relevance_result': context_relevance_result,
            'groundness_result': groundness_result,
            'answer_relevance_result': answer_relevance_result,
        }

    async def evaluate_context_relevance(self, query: str, docs: list[str]) -> ContextRelevanceResult:
        embeddings = await self.embed_model.aembed_documents([query] + docs)
        query_embed = embeddings[0]
        docs_embed = embeddings[1:]
        result = []
        for doc, doc_embed in zip(docs, docs_embed):
            score = self.cosine_similarity(query_embed, doc_embed)
            result.append(ContextRelevance(query=query, document=doc, score=score))

        relevant_count = sum(1 for r in result if r.score >= self.relevance_threshold)
        return ContextRelevanceResult(context_relevance=result, context_score=relevant_count / len(docs))

    @staticmethod
    def cosine_similarity(a, b):
        a = np.array(a)
        b = np.array(b)
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    async def evaluate_answer_relevance(self, query: str, answer: str) -> AnswerRelevanceResult:
        try:
            result: LLMAnswerRelevance = await self.llm_client.completions_create(
                system_prompt=answer_relevance_system_prompt,
                messages=[HumanMessage(content=answer_relevance_user_prompt.format(answer=answer))],
                response_class=LLMAnswerRelevance,
            )
            if not result.questions:
                return AnswerRelevanceResult(score=0.0, questions=[])
            embeddings = await self.embed_model.aembed_documents([query] + result.questions)
            query_embed = embeddings[0]
            sim_questions_embeds = embeddings[1:]
            scores = []
            for embedding, llm_question in zip(sim_questions_embeds, result.questions):
                score = self.cosine_similarity(query_embed, embedding)
                scores.append(score)

            score = round(np.mean(scores), 2) if scores else 0.0
            return AnswerRelevanceResult(questions=result.questions, score=score)
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(f"Answer relevance evaluation failed: {e}")
            return AnswerRelevanceResult(questions=[], score=0.0)


    async def evaluate_faithfulness(self, context: list[str], answer: str) -> GroundnessResult:
        try:
            user_prompt = evaluate_groundness_user_prompt.format(context=context, answer=answer)
            result: GroundnessResult = await self.llm_client.completions_create(
                system_prompt=evaluate_groundness_system_prompt,
                messages=[HumanMessage(content=user_prompt)],
                response_class=GroundnessResult,
            )
            return result
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(f"Faithfulness evaluation failed: {e}")
            return GroundnessResult(claims=[], faithfulness_score=0.0)
