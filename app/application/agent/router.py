from fastapi import APIRouter
from langchain_core.messages import HumanMessage
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel

from app.infra.db.repos.user_threads import UserThreadsRepo
from app.logic.nodes.state import MessagesState


class AgentRequest(BaseModel):
    question: str
    user_id: str
    thread_id: str | None = None


class AgentRouter:
    def __init__(
        self, graph_agent: CompiledStateGraph, user_threads_repo: UserThreadsRepo
    ):
        self.graph_agent = graph_agent
        self.user_threads_repo = user_threads_repo
        self.router = APIRouter(prefix="/agent", tags=["agent"])
        self.register_routes()

    def register_routes(self):
        self.router.post("/query")(self.agent_query)

    async def agent_query(self, payload: AgentRequest):
        config = {"configurable": {"thread_id": payload.thread_id}}
        res = await self.graph_agent.ainvoke(
            MessagesState(
                new_messages=[HumanMessage(content=payload.question)],
                question=payload.question,
            ),
            config=config,
        )
        # return res['messages'][-1].content
        return res
