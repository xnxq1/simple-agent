import uuid

from app.infra.qdrant.repos.interfaces import QdrantPoint
from app.infra.qdrant.repos.repos import QdrantRepo
from app.logic.nodes.base import BaseIngestNode
from app.logic.nodes.state import IngestState


class QdrantIngestNode(BaseIngestNode):
    def __init__(self, qdrant_repo: QdrantRepo):
        self.qdrant_repo = qdrant_repo

    async def execute(self, state: IngestState):
        points = [QdrantPoint(
                        vector={
                            "dense": embedding,
                        },
                        id=uuid.uuid4(),
                        payload={
                            **chunk.metadata,
                            "text": chunk.text,
                        },
                    ) for embedding, chunk in zip(state.embeddings, state.chunks)]
        await self.qdrant_repo.create_or_update_vector(collection_name='test', points=points)
