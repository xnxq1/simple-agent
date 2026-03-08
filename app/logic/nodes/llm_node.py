from app.infra.llm.client import LLMClient
from app.logic.nodes.base import BaseLLMNode
from app.logic.nodes.state import MessagesState

class LLMNode(BaseLLMNode):

    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

    async def execute(self, state: MessagesState) -> dict:
        result = await self.llm_client.completions_create(
            system_prompt="You are a helpful assistant with access to calculation tools. When solving multi-step problems, call tools one at a time. Use the result of each tool call as input for the next step. Do not call multiple dependent tools in parallel.",
            messages=state.messages
        )
        return {
            "messages": [result],
            "llm_calls": state.llm_calls + 1
        }