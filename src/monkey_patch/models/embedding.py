from typing import TypeVar, List, Generic
import numpy as np
from pydantic.generics import GenericModel

T = TypeVar('T')


class Embedding(GenericModel, Generic[T], T):
    def __init__(self, data: List[float], **kwargs):
        if issubclass(T, np.ndarray):
            super().__init__(np.array(data), **kwargs)
        else:
            super().__init__(data, **kwargs)
