import asyncio

from langchain_core.embeddings import Embeddings
from qdrant_client import models
from sentence_transformers import CrossEncoder

from app.infra.config import Settings
from app.infra.qdrant.repos.repos import QdrantRepo


class RAGTools:
    def __init__(
        self,
        qdrant_repo: QdrantRepo,
        embed_model: Embeddings,
        settings: Settings,
        cross_encoder_model: CrossEncoder,
    ):
        self.qdrant_repo = qdrant_repo
        self.embed_model = embed_model
        self.settings = settings
        self.cross_encoder_model = cross_encoder_model

    async def search_docs(
        self, text: str, topics: list[str] | None = None
    ) -> list[str]:
        """Search for documents similar to the given query text.

        Performs semantic search by embedding the query text and finding the most
        similar documents in the vector database using cosine similarity.

        Args:
            text: The search query string to find similar documents for.
            topics: Optional list of topics to filter results by. When provided, returns documents
                that have one of the specified topics OR documents with no topic assigned.

        Returns:
            QueryResponse: A Qdrant QueryResponse object containing matched documents
                with their IDs, scores, and metadata.

        Raises:
            CollectionNotExistError: If the target collection does not exist in Qdrant.

        Example:
            results = await rag_tools.search_docs("machine learning algorithms")
            for point in results.points:
                 print(f"Score: {point.score}, Text: {point.payload['text']}")
        """
        query_embed = await self.embed_model.aembed_query(text=text)
        query_filter = None
        top_k = 5
        if topics:
            # Filter by matching topics OR documents without a topic assigned
            query_filter = models.Filter(
                should=[
                    models.FieldCondition(
                        key="topic", match=models.MatchAny(any=topics)
                    ),
                    models.IsEmptyCondition(is_empty=models.PayloadField(key="topic")),
                ]
            )
        result = await self.qdrant_repo.search(
            collection_name=self.settings.qdrant_collection,
            vector=query_embed,
            limit=top_k * 5,
            query_filter=query_filter,
        )
        return await self.rerank(
            documents=[p.payload["text"] for p in result.points],
            query=text,
            limit=top_k,
        )

    async def rerank(
        self, documents: list[str], query: str, limit: int = 5
    ) -> list[str]:
        input = [[query, d] for d in documents]
        scores = await asyncio.to_thread(
            self.cross_encoder_model.predict, sentences=input
        )
        ranked = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in ranked[:limit]]
