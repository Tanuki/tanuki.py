# import abstract base class
from abc import ABC, abstractmethod
from typing import List

from tanuki.models.embedding import Embedding


class Embedding_API(ABC):
    def __init__(self) -> None:
        pass
        
    @abstractmethod
    def embed(self, texts: List[str], model: str = None, **kwargs) -> List[Embedding]:
        """
        The main embedding function, given the model and prompt, return a vector representation
        """
        pass