from langchain_core.messages import SystemMessage

from app.infra.llm.client import LLMClient
from app.logic.nodes.base import BaseLLMNode
from app.logic.nodes.state import MessagesState


class LLMNode(BaseLLMNode):
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

    async def execute(self, state: MessagesState) -> dict:
        messages = list(state.old_messages) + list(state.new_messages)
        if state.summary:
            messages = [
                SystemMessage(content=f"Резюме предыдущего диалога:\n{state.summary}")
            ] + messages
        result = await self.llm_client.completions_create(
            system_prompt="""Ты — агент поиска информации. Отвечаешь исключительно на основе данных из базы знаний.


Алгоритм:
1. Вызови search_docs с запросом по смыслу вопроса.
2. Если результатов недостаточно — повтори search_docs с другой формулировкой.
3. Сформируй ответ строго на основе найденных документов.

Правила:
- Используй ТОЛЬКО информацию из найденных документов.
- Если в документах нет ответа — сообщи: «В базе знаний нет информации по этому вопросу.»
- Не добавляй ничего от себя и не опирайся на собственные знания.""",
            messages=messages,
        )
        return {"new_messages": [result], "llm_calls": state.llm_calls + 1}
