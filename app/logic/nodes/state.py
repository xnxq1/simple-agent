from pydantic import BaseModel


class StateSchema(BaseModel):
    query: str
    result: str = None