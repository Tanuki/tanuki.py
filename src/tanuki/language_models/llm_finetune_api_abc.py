# import abstract base class
from abc import ABC, abstractmethod
from typing import List

from tanuki.models.finetune_job import FinetuneJob


class LLM_Finetune_API(ABC):
    def __init__(self) -> None:
        pass
        
    @abstractmethod  
    def list_finetuned(self, limit=100, **kwargs) -> List[FinetuneJob]:
        """
        Gets the last N fine-tuning runs
        """
        pass

    @abstractmethod
    def get_finetuned(self, job_id: str, **kwargs) -> FinetuneJob:
        """
        Gets a fine-tuning run by id
        """
        pass

    @abstractmethod
    def finetune(self, **kwargs) -> FinetuneJob:
        """
        Creates a fine-tuning run
        Args:
            **kwargs: 

        Returns:
        """
        pass


