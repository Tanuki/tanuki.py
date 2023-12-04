import abc 
from pydantic import BaseModel
class BaseModelConfig(abc.ABC, BaseModel):
    model_name: str
    provider: str
    context_length: int