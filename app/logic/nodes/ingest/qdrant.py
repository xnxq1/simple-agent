from app.infra.config import Settings
from app.infra.qdrant.repos.interfaces import QdrantPoint
from app.infra.qdrant.repos.repos import QdrantRepo
from app.logic.nodes.ingest.base import BaseIngestNode, IngestState


class QdrantIngestNode(BaseIngestNode):
    def __init__(self, qdrant_repo: QdrantRepo, settings: Settings):
        self.qdrant_repo = qdrant_repo
        self.settings = settings

    async def execute(self, state: IngestState) -> dict:
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
            for embedding, chunk in zip(state.embeddings, state.chunks)
        ]
        await self.qdrant_repo.create_or_update_vector(
            collection_name=self.settings.qdrant_collection, points=points
        )
        return {}
