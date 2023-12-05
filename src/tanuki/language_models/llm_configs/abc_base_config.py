import abc 
from pydantic import BaseModel

class BaseModelConfig(abc.ABC, BaseModel):
    model_name: str
    provider: str
    context_length: int
    chat_template : str = None
    system_message : str = None
    instructions : str = None

    @property
    def instructions(self):
        return self._instructions
    
    @property
    def system_message(self):
        return self._system_message
    
    @property
    def chat_template(self):
        return self._chat_template