from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

from app.infra.db.repos.query_traces import QueryTracesRepo
from app.logic.nodes.state import MessagesState


class SaveEpisodeNode:
    def __init__(self, query_traces_repo: QueryTracesRepo) -> None:
        self.query_traces_repo = query_traces_repo

    async def execute(self, state: MessagesState, config: RunnableConfig) -> dict:
        thread_id = config["configurable"].get("thread_id")
        if not thread_id:
            return {}

        tools_used: list[str] = []
        topics: list[str] = []
        for msg in state.new_messages:
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    tools_used.append(tc["name"])
                    if tc["name"] == "search_docs":
                        msg_topics = (tc.get("args") or {}).get("topics") or []
                        topics.extend(msg_topics)

        human_msgs = [m for m in state.new_messages if isinstance(m, HumanMessage)]
        message_id = human_msgs[-1].id if human_msgs else ""

        await self.query_traces_repo.insert(
            payload={
                "thread_id": thread_id,
                "message_id": message_id,
                "question": state.question,
                "answer": state.answer or "",
                "tools_used": list(set(tools_used)),
                "topics": list(set(topics)),
                "context_score": state.context_relevance_result.context_score
                if state.context_relevance_result
                else None,
                "faithfulness_score": state.groundness_result.faithfulness_score
                if state.groundness_result
                else None,
                "answer_relevance_score": state.answer_relevance_result.score
                if state.answer_relevance_result
                else None,
            }
        )
        return {}
