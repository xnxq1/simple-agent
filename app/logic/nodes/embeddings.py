import asyncio

from langchain_huggingface import HuggingFaceEmbeddings

from app.logic.nodes.base import BaseIngestNode
from app.logic.nodes.state import IngestState


class EmbeddingNode(BaseIngestNode):
    def __init__(self, embed_model):
        self.embed_model = embed_model
    async def execute(self, state: IngestState):
        embeddings = await asyncio.to_thread(self.embed_model.embed_documents, texts=[chunk.text for chunk in state.chunks])
        return {'embeddings': embeddings}