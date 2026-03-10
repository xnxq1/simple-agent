import asyncio
import re
import uuid

from langchain_text_splitters import RecursiveCharacterTextSplitter
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from app.logic.nodes.ingest.base import BaseIngestNode, Chunk, IngestState, SemanticChunk

MIN_CHUNK_LENGTH = 50


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

    @staticmethod
    def _is_garbage(text: str) -> bool:
        """Check if chunk is too short or contains no alphabetic characters."""
        if len(text.strip()) < MIN_CHUNK_LENGTH:
            return True
        return not re.search(r'[a-zA-Zа-яА-Я]', text)

    async def execute(self, state: IngestState) -> dict:
        semantic_nodes = await self.semantic_splitter.aget_nodes_from_documents(
            documents=state.documents
        )
        final_chunks = []
        semantic_chunks = []
        for node in semantic_nodes:
            metadata = node.metadata
            chunks = await asyncio.to_thread(self.sentence_splitter.split_text, text=node.text)
            chunks = [t for t in chunks if not self._is_garbage(t)]
            if not chunks:
                continue
            chunks = [Chunk(id=uuid.uuid4(), text=chunk, metadata=metadata) for chunk in chunks]
            final_chunks.extend(chunks)
            semantic_chunks.append(SemanticChunk(text=node.text, metadata={}, chunk_ids=[chunk.id for chunk in chunks]))
        return {"chunks": final_chunks, "semantic_chunks": semantic_chunks}