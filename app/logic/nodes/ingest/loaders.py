import logging
from llama_index.core.readers.base import BaseReader

from app.logic.nodes.ingest.base import BaseIngestNode, IngestState

logger = logging.getLogger(__name__)


class WebLoaderNode(BaseIngestNode):
    def __init__(self, reader: BaseReader):
        self.reader = reader


    async def execute(self, state: IngestState) -> dict:
        documents = []
        failed_urls = []

        for url in state.urls:
            try:
                docs = await self.reader.aload_data(urls=[url])
                # Ensure URL is in metadata for all documents
                for doc in docs:
                    if 'url' not in doc.metadata:
                        doc.metadata['url'] = url
                documents.extend(docs)
            except Exception as e:
                logger.warning(f"Failed to load URL {url}: {e}")
                failed_urls.append(url)

        if failed_urls:
            logger.warning(f"Failed to load {len(failed_urls)} URLs: {failed_urls}")

        if not documents:
            logger.warning("No documents loaded from any of the provided URLs")

        return {'documents': documents}