import asyncio
import json
import logging
from typing import Callable

from langchain_core.messages import ToolMessage

from app.logic.nodes.state import MessagesState
from app.logic.tools.rag import RAGTools

logger = logging.getLogger(__name__)


class ToolNode:
    def __init__(self, tools: list[Callable]):
        # Build a name → callable registry from tool functions
        self._tools: dict[str, Callable] = {tool.__name__: tool for tool in tools}

    async def execute(self, state: MessagesState) -> dict:
        last_message = state.messages[-1]
        tool_calls = getattr(last_message, "tool_calls", [])

        # Run all tool calls in parallel, capturing individual exceptions
        results = await asyncio.gather(
            *[self._call_tool(tc) for tc in tool_calls],
            return_exceptions=True,
        )

        tool_messages = []
        for tool_call, result in zip(tool_calls, results):
            if isinstance(result, Exception):
                logger.error("Tool %s failed: %s", tool_call["name"], result)
                content = f"Error: {result}"
            else:
                logger.debug("Tool %s returned: %s", tool_call["name"], result)
                print(tool_call["name"], RAGTools.search_docs.__name__)
                print('\n\n\n\n\n\n')
                if tool_call["name"] == RAGTools.search_docs.__name__:
                    state.retrieve_context.extend(
                        [point.payload["text"] for point in result.points]
                    )
                    print(state.retrieve_context)
                    print('\n\n\n\n\n\n')
                content = json.dumps(result, default=str)

            tool_messages.append(
                ToolMessage(
                    content=content,
                    tool_call_id=tool_call["id"],
                    name=tool_call["name"],
                )
            )

        return {"messages": tool_messages}

    async def _call_tool(self, tool_call: dict):
        name = tool_call["name"]
        args = tool_call.get("args", {})

        tool = self._tools.get(name)
        if tool is None:
            raise ValueError(f"Unknown tool: {name}")

        logger.debug("Calling tool %s with args %s", name, args)
        return await tool(**args)
