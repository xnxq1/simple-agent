from langchain_core.messages import AIMessage

from app.infra.llm.client import LLMClient
from app.logic.nodes.base import Node
from app.logic.nodes.state import MessagesState

class LLMNode(Node):

    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

    async def execute(self, state: MessagesState) -> dict:
        last_message = state["messages"][-1] if state["messages"] else None
        user_query = last_message.content if last_message else ""

        result = await self.llm_client.completions_create(
            system_prompt="You are a helpful assistant. When you need to perform calculations, use the available tools.",
            user_query=user_query
        )

        return {
            "messages": [AIMessage(content=result)],
            "llm_calls": state["llm_calls"] + 1
        }