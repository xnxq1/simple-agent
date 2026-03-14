"""Service for persisting chunks to vector database."""

from app.infra.config import Settings
from app.infra.qdrant.repos.interfaces import QdrantPoint
from app.infra.qdrant.repos.repos import QdrantRepo
from app.logic.nodes.ingest.base import Chunk


class VectorStoreService:
    """Handles construction and persistence of chunks to Qdrant."""

    def __init__(self, qdrant_repo: QdrantRepo, settings: Settings):
        self.qdrant_repo = qdrant_repo
        self.settings = settings

    async def upsert_chunks(
        self, chunks: list[Chunk], embeddings: list[list[float]]
    ) -> None:
        """Persist chunks with embeddings to Qdrant.

        Args:
            chunks: List of chunks to persist
            embeddings: List of embedding vectors (same length as chunks)
        """
        points = [
            QdrantPoint(
                vector={
                    "dense": embedding,
                },
                id=chunk.id,
                payload={
                    **chunk.metadata,
                    "text": chunk.text,
                },
            )
            for embedding, chunk in zip(embeddings, chunks)
        ]
        await self.qdrant_repo.create_or_update_vector(
            collection_name=self.settings.qdrant_collection, points=points
        )
