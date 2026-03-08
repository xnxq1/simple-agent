from app.logic.nodes.base import BaseIngestNode
from llama_index.readers.web import TrafilaturaWebReader

from app.logic.nodes.state import IngestState


class WebLoaderNode(BaseIngestNode):
    def __init__(self):
        ...


    async def execute(self, state: IngestState):
        reader = TrafilaturaWebReader()

        documents = reader.load_data(
            urls=[state.url],
        )
        return {'docs': documents}