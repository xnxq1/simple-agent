from abc import ABC, abstractmethod

from qdrant_client.conversions.common_types import PointStruct
from qdrant_client.http.models import QueryResponse


class QdrantPoint(PointStruct): ...


class QdrantInterface(ABC):
    @abstractmethod
    async def create_collection(self, collection_name: str, size: int) -> None: ...

    @abstractmethod
    async def create_or_update_vector(self, collection_name: str, points: list[QdrantPoint]) -> None: ...

    @abstractmethod
    async def search(self, collection_name: str, vector: list[float], limit: int | None = None, query_filter=None) -> QueryResponse: ...

    @abstractmethod
    async def hybrid_search(self, collection_name: str, vector: list[float], sparse_vector: dict, limit: int | None = None) -> QueryResponse: ...
