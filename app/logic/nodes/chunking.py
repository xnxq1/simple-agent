import asyncio

from langchain_text_splitters import RecursiveCharacterTextSplitter
from llama_index.core.node_parser import SemanticSplitterNodeParser, SentenceSplitter
from llama_index.core import Document
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from app.logic.nodes.base import BaseIngestNode
from app.logic.nodes.state import IngestState, Chunk


class SemanticChunkingNode(BaseIngestNode):
    def __init__(self):
        self.semantic_splitter = SemanticSplitterNodeParser(
            embed_model=HuggingFaceEmbedding(),
            breakpoint_percentile_threshold=95
        )
        self.sentence_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
        )

    async def execute(self, state: IngestState) -> dict:
        semantic_nodes = await self.semantic_splitter.aget_nodes_from_documents(
            documents=state.documents
        )
        final_chunks = []
        for node in semantic_nodes:
            metadata = node.metadata
            chunks = await asyncio.to_thread(self.sentence_splitter.split_text, text=node.text)
            final_chunks.extend([Chunk(text=chunk, metadata=metadata) for chunk in chunks])
        return {"chunks": final_chunks}