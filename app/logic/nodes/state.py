from pydantic import BaseModel

from app.infra.models import Diagram


class StateSchema(BaseModel):
    query: str
    result: Diagram = None
    image: str = None
