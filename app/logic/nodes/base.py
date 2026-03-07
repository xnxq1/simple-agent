import abc

from app.logic.nodes.state import MessagesState


class Node(abc.ABC):

    @abc.abstractmethod
    def execute(self, state: MessagesState) -> MessagesState:
        ...