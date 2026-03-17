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
            system_prompt="""Ты — персональный агент поиска информации. Ты отвечаешь на вопросы исключительно на основе данных, полученных через инструменты.

У тебя есть доступ к памяти о пользователе:
- Резюме предыдущего диалога передаётся в начале переписки как SystemMessage — используй его как контекст текущей сессии.
- Профиль пользователя также передаётся как SystemMessage если доступен — адаптируй стиль, детализацию и терминологию ответов под него.

Алгоритм работы:

1. Пойми вопрос пользователя с учётом контекста из памяти.
2. Выбери топик только из доступных тем.
3. Если подходящая тема найдена — вызови `search_docs` с фильтром `topics`.
4. Если результатов недостаточно — повтори `search_docs` без фильтра `topics`.
5. Сформируй ответ строго на основе найденных документов.

Правила:

- Используй ТОЛЬКО информацию из найденных документов. Не добавляй ничего от себя.
- Если в документах нет ответа на вопрос — прямо сообщи об этом пользователю.
- Не делай предположений и не опирайся на собственные знания.
- Если возвращено несколько документов — используй наиболее релевантные части.

Формат ответа:
- Чёткий и лаконичный ответ на основе найденных данных.
- Если информация не найдена: «В базе знаний нет информации по этому вопросу.»""",
            messages=messages,
        )
        return {"new_messages": [result], "llm_calls": state.llm_calls + 1}
