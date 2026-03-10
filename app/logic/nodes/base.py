import abc

from app.logic.nodes.state import MessagesState


class BaseLLMNode(abc.ABC):

    @abc.abstractmethod
    async def execute(self, state: MessagesState) -> MessagesState:
        ...