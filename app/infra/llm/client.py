import json
from logging import getLogger
from typing import Any

from openai import AsyncOpenAI
from openai.types.chat import (
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)

from app.infra.config import Settings

logger = getLogger(__name__)


class LLMClient:
    def __init__(
        self,
        client: AsyncOpenAI,
        settings: Settings,
    ):
        self.client = client
        self.settings = settings

    async def completions_create(
        self,
        system_prompt: str,
        user_query: str,
        response_format: dict | None = None,
        response_class: type | None = None,
    ) -> Any:
        chat_settings = {}
        if response_format:
            chat_settings["response_format"] = response_format
        result = await self.client.chat.completions.create(
            model=self.settings.llm_model,
            temperature=0.2,
            messages=[
                ChatCompletionSystemMessageParam(content=system_prompt, role="system"),
                ChatCompletionUserMessageParam(content=user_query, role="user"),
            ],
            **chat_settings,
        )
        total_tokens = result.usage.total_tokens
        input_tokens = result.usage.prompt_tokens
        output_tokens = result.usage.completion_tokens
        logger.info(
            f"Query to {self.settings.llm_model} "
            f"input_tokens: {input_tokens}, output_tokens: {output_tokens}, total_tokens: {total_tokens} "
        )
        if response_format:
            res = json.loads(result.choices[0].message.content)
            if response_class:
                return response_class(**res)
            return res

        return result.choices[0].message.content
