import abc 
from pydantic import BaseModel, ConfigDict
from typing import Optional

class BaseModelConfig(abc.ABC, BaseModel):
    """
    The abstract Basemodel class for all model configs.

    Parameters
    ----------
    model_name : str -- the name of the model
    provider : str -- the name of the provider (api provider)
    context_length : int -- the context length of the model
    chat_template : Optional[str] -- the chat template for the model
    system_message : Optional[str] -- the system message for the model
    instructions : Optional[str] -- the instructions for the model
    parsing_helper_tokens : Optional[dict] -- the parsing helper tokens for the model
    """
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