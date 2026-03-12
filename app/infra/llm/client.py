import json
from logging import getLogger
from typing import Any, Callable

from langchain_core.messages import BaseMessage, SystemMessage
from langchain_openai import ChatOpenAI

from app.infra.config import Settings

logger = getLogger(__name__)


class LLMClient:
    def __init__(
        self,
        model: ChatOpenAI,
        settings: Settings,
        tools: list[Callable] | None = None,
    ):
        self.model = model
        self.settings = settings
        self.tools = tools

        if tools:
            self.model = self.model.bind_tools(tools)

    async def completions_create(
        self,
        system_prompt: str,
        messages: list[BaseMessage],
        response_format: dict | None = None,
        response_class: type | None = None,
    ) -> BaseMessage | Any:
        full_messages = [SystemMessage(content=system_prompt)] + messages

        if response_format or response_class:
            if response_class:
                model = self.model.with_structured_output(response_class)
                result = await model.ainvoke(full_messages)
                logger.info(
                    f"Query to {self.settings.llm_model} with structured output"
                )
                return result
            else:
                result = await self.model.ainvoke(full_messages)
                logger.info(f"Query to {self.settings.llm_model}")
                res = json.loads(result.content)
                return res

        result = await self.model.ainvoke(full_messages)
        logger.info(f"Query to {self.settings.llm_model}")
        return result
