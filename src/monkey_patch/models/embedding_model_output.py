from dataclasses import dataclass
from typing import TypeVar

from monkey_patch.models.embedding import Embedding

T = TypeVar('T')

@dataclass()
class EmbeddingModelOutput:
    generated_response: Embedding[T]
    #suitable_for_finetuning: bool
    distilled_model: bool
