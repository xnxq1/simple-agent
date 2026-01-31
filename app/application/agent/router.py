from typing import TypedDict

from fastapi import APIRouter
from langgraph.graph import StateGraph
from pydantic import BaseModel

router = APIRouter(prefix="/agent", tags=["agent"])

@router.post('/query')
async def agent_query(query: str):
    ...