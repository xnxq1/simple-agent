from llama_index.core.readers.base import BaseReader

from app.logic.nodes.ingest.base import BaseIngestNode, IngestState



class WebLoaderNode(BaseIngestNode):
    def __init__(self, reader: BaseReader):
        self.reader = reader


    async def execute(self, state: IngestState):
        documents = await self.reader.aload_data(
            urls=state.urls,
        )
        return {'documents': documents}