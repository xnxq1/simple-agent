from pydantic import BaseModel


class StateSchema(BaseModel):
    query: str