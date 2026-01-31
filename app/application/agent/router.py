from typing import TypedDict

from fastapi import APIRouter, FastAPI
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel

from app.logic.nodes.state import StateSchema


class AgentRouter:
    def __init__(self, graph_agent: CompiledStateGraph):
        self.graph_agent = graph_agent
        self.router = APIRouter(prefix="/agent", tags=["agent"])
        self.register_routes()

    def register_routes(self):
        self.router.post("/query")(self.agent_query)

    async def agent_query(self, payload: str):
        res= await self.graph_agent.ainvoke(StateSchema(query=payload))
        return res['result']