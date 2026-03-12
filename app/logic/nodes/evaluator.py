import abc

from langchain_core.embeddings import Embeddings

from app.infra.llm.client import LLMClient
from app.logic.nodes.base import BaseLLMNode
from app.logic.nodes.state import MessagesState


class Evaluator(BaseLLMNode):
    def __init__(self, llm_client: LLMClient, embed_model: Embeddings):
        self.llm_client = llm_client
        self.embed_model = embed_model

    @abc.abstractmethod
    async def execute(self, state: MessagesState) -> MessagesState: ...

    async def evaluate_context_relevance(self): ...

    async def evaluate_answer_relevance(self): ...

    async def evaluate_groundness(self): ...
