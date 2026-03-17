from langchain_core.messages import RemoveMessage
from langchain_core.runnables import RunnableConfig

from app.infra.db.repos.query_traces import QueryTracesRepo
from app.infra.db.repos.thread_summaries import ThreadSummariesRepo
from app.logic.nodes.state import MessagesState


class LoadSummaryNode:
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

        summary = await self.thread_summaries_repo.get_latest(thread_id)
        if not summary:
            return {}

        covered_traces = await self.query_traces_repo.get_by_summary_id(summary.id)
        covered_message_ids = {t.message_id for t in covered_traces}

        remove_messages = [
            RemoveMessage(id=mid)
            for mid in covered_message_ids
            if any(m.id == mid for m in state.messages)
        ]

        result: dict = {
            "summary": summary.summary,
            "summary_id": str(summary.id),
        }
        if remove_messages:
            result["messages"] = remove_messages

        return result
