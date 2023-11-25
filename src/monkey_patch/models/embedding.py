from typing import TypeVar, List, Generic, Union, get_args, get_origin

import numpy as np

T = TypeVar('T')


class Embedding(Generic[T]):
    _data_type = None  # Placeholder for the data type

    def __init__(self, data: List[float]):
        # Determine the origin of the data type (list, np.ndarray, etc.)
        data_type_origin = get_origin(self._data_type) or self._data_type

        if data_type_origin is np.ndarray:
            self._data = np.array(data)
        elif data_type_origin is list:
            # Further check for element type if necessary
            element_type = get_args(self._data_type)[0] if get_args(self._data_type) else None
            if all(isinstance(item, element_type) for item in data):
                self._data = data
            else:
                raise TypeError("Elements of data do not match the expected type")
        elif not data_type_origin:
            self._data = data
        else:
            raise TypeError("Unsupported data type for embedding")

    @classmethod
    def __class_getitem__(cls, item):
        new_cls = type(cls.__name__, (cls,), {'_data_type': item})
        return new_cls

    def __getattr__(self, item):
        return getattr(self._data, item)

    def __repr__(self):
        return repr(self._data)

    def __str__(self):
        return str(self._data)