from pydantic import BaseModel
from datetime import datetime
from typing import List

class TodoItem(BaseModel):
    deadline: datetime
    goal: str
    people: List[str]