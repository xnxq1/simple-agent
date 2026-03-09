
from langchain_core.embeddings import Embeddings
from qdrant_client.http.models import QueryResponse

from app.infra.qdrant.repos.repos import QdrantRepo


class RAGTools:
    def __init__(
        self,
        qdrant_repo: QdrantRepo,
        embed_model: Embeddings,
    ):
        self.qdrant_repo = qdrant_repo
        self.embed_model = embed_model

    async def search_docs(self, text: str, top_k: int) -> QueryResponse:
        """Search for documents similar to the given query text.

        Performs semantic search by embedding the query text and finding the most
        similar documents in the vector database using cosine similarity.

        Args:
            text: The search query string to find similar documents for.
            top_k: Maximum number of similar documents to return (1-100).

        Returns:
            QueryResponse: A Qdrant QueryResponse object containing matched documents
                with their IDs, scores, and metadata.

        Raises:
            CollectionNotExistError: If the target collection does not exist in Qdrant.

        Example:
            results = await rag_tools.search_docs("machine learning algorithms", top_k=5)
            for point in results.points:
                 print(f"Score: {point.score}, Text: {point.payload['text']}")
        """
        query_embed = await self.embed_model.aembed_query(text=text)
        return await self.qdrant_repo.search(collection_name='test', vector=query_embed, limit=top_k)


