import abc

from app.logic.nodes.state import MessagesState, IngestState


class BaseLLMNode(abc.ABC):

    @abc.abstractmethod
    async def execute(self, state: MessagesState) -> MessagesState:
        ...

class BaseIngestNode(abc.ABC):

    @abc.abstractmethod
    async def execute(self, state: IngestState) -> None:
        ...