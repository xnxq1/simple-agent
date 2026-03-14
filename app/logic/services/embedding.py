"""Service for embedding chunks into vectors."""

from langchain_core.embeddings import Embeddings

from app.logic.nodes.ingest.base import Chunk


class EmbeddingService:
    """Handles batch embedding of text chunks."""

    def __init__(self, embed_model: Embeddings):
        self.embed_model = embed_model

    async def embed_chunks(self, chunks: list[Chunk]) -> list[list[float]]:
        """Embed a list of chunks into vectors.

        Args:
            chunks: List of chunks to embed

        Returns:
            List of embedding vectors, one per chunk
        """
        embeddings = await self.embed_model.aembed_documents(
            texts=[chunk.text for chunk in chunks]
        )
        return embeddings
