from llama_index.core.readers.base import BaseReader

from app.logic.nodes.base import BaseIngestNode
from llama_index.readers.web import TrafilaturaWebReader

from app.logic.nodes.state import IngestState


class WebLoaderNode(BaseIngestNode):
    def __init__(self, reader: BaseReader):
        self.reader = reader


    async def execute(self, state: IngestState):
        documents = self.reader.load_data(
            urls=state.urls,
        )
        return {'docs': documents}