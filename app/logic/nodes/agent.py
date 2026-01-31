from app.infra.llm.client import LLMClient
from app.logic.nodes.state import StateSchema


class AgentNode:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    async def execute(self, state: StateSchema) -> dict:
        res = await self.llm.completions_create(
            system_prompt="", user_query=state.query
        )
        return {"result": res}
