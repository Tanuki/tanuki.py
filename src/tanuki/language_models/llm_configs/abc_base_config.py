import abc 
from pydantic import BaseModel, ConfigDict
from typing import Optional

class BaseModelConfig(abc.ABC, BaseModel):
    model_config = ConfigDict(
            protected_namespaces=()
        )
    model_name: str
    provider: str
    context_length: int
    chat_template : Optional[str] = None
    system_message : Optional[str] = None
    instructions : Optional[str] = None
    parsing_helper_tokens: Optional[dict] = {"start_token": "", "end_token": ""}