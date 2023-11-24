# import abstract base class
from abc import ABC, abstractmethod


class LLM_API(ABC):
    def __init__(self) -> None:
        pass
        
    @abstractmethod  
    def generate(self, model, system_message, prompt, **kwargs):
        """
        The main generation function, given the args, kwargs, function_modeler, function description and model type, generate a response and check if the datapoint can be saved to the finetune dataset
        """
        pass