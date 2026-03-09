import asyncio

from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings

from app.logic.nodes.base import BaseIngestNode
from app.logic.nodes.state import IngestState


class EmbeddingNode(BaseIngestNode):
    def __init__(self, embed_model: Embeddings):
        self.embed_model = embed_model
    async def execute(self, state: IngestState):
        embeddings = await self.embed_model.aembed_documents(texts=[chunk.text for chunk in state.chunks])
        return {'embeddings': embeddings}