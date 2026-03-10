from app.infra.llm.client import LLMClient
from app.logic.nodes.base import BaseLLMNode
from app.logic.nodes.state import MessagesState

class LLMNode(BaseLLMNode):

    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

    async def execute(self, state: MessagesState) -> dict:
        result = await self.llm_client.completions_create(
            system_prompt="""
            You are an AI retrieval agent.

            You answer questions by retrieving information from tools.
            
            Workflow:
            
            1. Understand the user query.
            2. Decide if information retrieval is required.
            3. If yes, call the most appropriate tool.
            4. Review the retrieved documents carefully.
            5. Use the retrieved information to construct the answer.
            
            Guidelines:

            - Prefer `search_docs` for domain-specific information.
            - If multiple documents are returned, focus on the most relevant parts.
            - Do not invent information not present in the retrieved context.
            - If no relevant information is found, say so.
            
            Answer format:
            - Provide a clear and concise answer.
            - Use the retrieved information as the primary source.
            """,
            messages=state.messages
        )
        return {
            "messages": [result],
            "llm_calls": state.llm_calls + 1
        }