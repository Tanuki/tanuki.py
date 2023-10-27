from pydantic import BaseModel
from datetime import datetime
from typing import List, Optional


class TodoItem(BaseModel):
    deadline: Optional[datetime] = None
    goal: str
    people: List[str]