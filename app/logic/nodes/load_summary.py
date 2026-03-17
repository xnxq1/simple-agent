from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig

from app.infra.db.repos.query_traces import QueryTracesRepo
from app.infra.db.repos.thread_summaries import ThreadSummariesRepo
from app.logic.nodes.state import MessagesState


class LoadMemoryNode:
    def __init__(
        self,
        thread_summaries_repo: ThreadSummariesRepo,
        query_traces_repo: QueryTracesRepo,
    ) -> None:
        self.thread_summaries_repo = thread_summaries_repo
        self.query_traces_repo = query_traces_repo

    async def execute(self, state: MessagesState, config: RunnableConfig) -> dict:
        thread_id = config["configurable"].get("thread_id")
        if not thread_id:
            return {}

        result = {}

        summary = await self.thread_summaries_repo.get_latest(thread_id)
        if summary:
            result["summary"] = summary.summary
            result["summary_id"] = str(summary.id)

        unsummarized = await self.query_traces_repo.get_unsummarized(thread_id)
        result["old_messages"] = [
            msg
            for trace in unsummarized
            for msg in [
                HumanMessage(content=trace.question),
                AIMessage(content=trace.answer),
            ]
        ]

        return result
