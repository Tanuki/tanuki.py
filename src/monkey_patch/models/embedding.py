from typing import TypeVar, List, Generic, ClassVar, Union
import numpy as np
from pydantic.generics import GenericModel

T = TypeVar('T')


class Embedding(GenericModel, Generic[T]):
    def __init__(self, data: Union[List[float], np.ndarray]):
        object.__setattr__(self, '_data', data)

    def __getattribute__(self, item):
        # First, try to get attribute from the class itself
        try:
            return super().__getattribute__(item)
        except AttributeError:
            pass

        # Then, delegate to _data if available
        _data = super().__getattribute__('_data')
        if hasattr(_data, item):
            return getattr(_data, item)

        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{item}'")
