from typing import TypeVar, List, Generic
import numpy as np
from pydantic.generics import GenericModel

T = TypeVar('T')


class Embedding(GenericModel, Generic[T]):
    def __init__(self, data: List[float]):
        if issubclass(self.__class__.__orig_bases__[0].__args__[0], np.ndarray):
            self._data = np.array(data)
        else:
            self._data = data

    def __getattr__(self, item):
        return getattr(self._data, item)

    def __len__(self):
        return len(self._data)

    def __getitem__(self, key):
        return self._data[key]