# import abstract base class
from abc import ABC, abstractmethod
from typing import List

from tanuki.models.embedding import Embedding
from tanuki.language_models.llm_configs.abc_base_config import BaseModelConfig

class Embedding_API(ABC):
    def __init__(self) -> None:
        pass
        
    @abstractmethod
    def embed(self, texts: List[str], model: BaseModelConfig = None, **kwargs) -> List[Embedding]:
        """
        The main embedding function, given the model and prompt, return a vector representation
        """
        pass