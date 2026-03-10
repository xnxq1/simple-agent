import logging
from qdrant_client import AsyncQdrantClient, models
from qdrant_client.http.models import Distance, QueryResponse, VectorParams

from app.infra.config import Settings
from app.infra.qdrant.repos.exceptions import CollectionNotExistError
from app.infra.qdrant.repos.interfaces import QdrantInterface, QdrantPoint

logger = logging.getLogger(__name__)


class QdrantRepo(QdrantInterface):
    # Vector dimension for all-MiniLM-L6-v2 model
    EMBEDDING_DIMENSION = 384
    DEFAULT_COLLECTION = "test"

    def __init__(self, client: AsyncQdrantClient, settings: Settings):
        self.client = client
        self.settings = settings

    async def create_collection(self, collection_name: str, size: int) -> None:
        if not await self.client.collection_exists(collection_name=collection_name):
            await self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=size, distance=Distance.COSINE),
            )

    async def ensure_collection(self, collection_name: str = None) -> None:
        """Ensure the default or specified collection exists."""
        collection = collection_name or self.DEFAULT_COLLECTION
        if not await self.client.collection_exists(collection_name=collection):
            logger.info(f"Creating Qdrant collection '{collection}'")
            await self.create_collection(collection, self.EMBEDDING_DIMENSION)
        else:
            logger.debug(f"Qdrant collection '{collection}' already exists")

    async def create_or_update_vector(
        self, collection_name: str, points: list[QdrantPoint]
    ) -> None:
        if not await self.client.collection_exists(collection_name=collection_name):
            raise CollectionNotExistError(f"Collection {collection_name} does not exist")
        await self.client.upsert(
            collection_name=collection_name,
            points=points,
        )

    async def search(
        self, collection_name, vector, limit: int | None = None, query_filter=None
    ) -> QueryResponse:
        limit = limit or self.settings.top_k_limit
        return await self.client.query_points(
            collection_name=collection_name,
            query=vector,
            limit=limit,
            using='dense',
            query_filter=query_filter,
            score_threshold=self.settings.qdrant_score_threshold,
        )

    async def hybrid_search(
        self, collection_name, vector, sparse_vector, limit: int | None = None
    ) -> QueryResponse:
        limit = limit or self.settings.top_k_limit

        return await self.client.query_points(
            collection_name=collection_name,
            prefetch=[
                models.Prefetch(query=vector, using="dense", limit=20),
                models.Prefetch(
                    query=models.SparseVector(
                        indices=sparse_vector["indices"].tolist(),
                        values=sparse_vector["values"].tolist(),
                    ),
                    using="sparse",
                    limit=20,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=limit,
        )
