"""Service for loading and processing documents from URLs."""

import logging

from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document

logger = logging.getLogger(__name__)


class WebLoaderService:
    """Handles document loading from URLs with error handling and metadata enrichment."""

    def __init__(self, reader: BaseReader):
        self.reader = reader

    async def load_urls(self, urls: list[str]) -> list[Document]:
        """Load documents from URLs, guaranteeing 'url' in metadata.

        Args:
            urls: List of URLs to load

        Returns:
            List of documents with 'url' field guaranteed in metadata
        """
        documents = []
        failed_urls = []

        for url in urls:
            try:
                docs = await self.reader.aload_data(urls=[url])
                for doc in docs:
                    if "url" not in doc.metadata:
                        doc.metadata["url"] = url
                documents.extend(docs)
            except Exception as e:
                logger.warning(f"Failed to load URL {url}: {e}")
                failed_urls.append(url)

        if failed_urls:
            logger.warning(f"Failed to load {len(failed_urls)} URLs: {failed_urls}")

        if not documents:
            logger.warning("No documents loaded from any of the provided URLs")

        return documents
