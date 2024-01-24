from dataclasses import dataclass
from typing import List, Tuple, Set, Dict, Mapping, MutableMapping, OrderedDict, ChainMap, Counter, DefaultDict, Deque, \
    MutableSequence, Sequence, Union, Literal

from pydantic import BaseModel

from tanuki.validator import Validator



def test_validate_base_types():
    print("test_validate_base_types")
    validator = Validator()
    assert validator.check_type(1, int)
    assert validator.check_type(1.0, float)
    assert validator.check_type("1", str)
    assert validator.check_type(True, bool)
    assert validator.check_type(None, None)
    assert not validator.check_type(1, float)
    assert not validator.check_type(1.0, int)
    assert not validator.check_type("1", int)
    #assert not validator.validate_value(True, int)
    assert not validator.check_type(None, int)
    assert not validator.check_type(1, str)
    assert not validator.check_type(1.0, str)
    assert not validator.check_type(True, str)
    assert not validator.check_type(None, str)
    assert not validator.check_type(1, bool)
    assert not validator.check_type(1.0, bool)
    assert not validator.check_type("1", bool)
    assert not validator.check_type(None, bool)
    assert not validator.check_type(1, None)
    assert not validator.check_type(1.0, None)
    assert not validator.check_type("1", None)
    assert not validator.check_type(True, None)

def test_validate_complex_types():
    print("test_validate_complex_types")
    validator = Validator()
    assert validator.check_type([1, 2, 3], list)
    assert validator.check_type([1, 2, 3], list[int])
    assert not validator.check_type([1, 2, 3], list[str])
    assert validator.check_type((1, 2, 3), tuple)
    assert validator.check_type({1, 2, 3}, set)
    assert validator.check_type({"a": 1, "b": 2, "c": 3}, dict)
    assert not validator.check_type([1, 2, 3], tuple)
    assert not validator.check_type((1, 2, 3), list)
    assert not validator.check_type({1, 2, 3}, dict)
    assert not validator.check_type({"a": 1, "b": 2, "c": 3}, set)
    assert not validator.check_type([1, 2, 3], dict)
    assert not validator.check_type((1, 2, 3), set)
    assert not validator.check_type({1, 2, 3}, list)
    assert not validator.check_type({"a": 1, "b": 2, "c": 3}, tuple)
    assert not validator.check_type([1, 2, 3], None)
    assert not validator.check_type((1, 2, 3), None)


def test_validate_type_annotations():
    print("test_validate_type_annotations")
    validator = Validator()
    assert validator.check_type([1, 2], List[int])
    assert validator.check_type([1, 2], List)
    assert not validator.check_type([1, 2], List[str])

    assert validator.check_type((1, 2), Tuple[int, int])
    assert validator.check_type((1, 2), Tuple)
    assert not validator.check_type((1, 2), Tuple[str, str])

    assert validator.check_type({1, 2}, Set[int])
    assert validator.check_type({1, 2}, Set)
    assert not validator.check_type({1, 2}, Set[str])

    assert validator.check_type({"a": 1, "b": 2}, Dict[str, int])
    assert validator.check_type({"a": 1, "b": 2}, Dict)
    assert not validator.check_type({"a": 1, "b": 2}, Dict[str, str])

def test_validate_collection_dict_types():
    print("test_validate_collection_types")
    validator = Validator()
    assert validator.check_type({"a": 1, "b": 2}, OrderedDict[str, int])
    assert validator.check_type({"a": 1, "b": 2}, OrderedDict)

    assert validator.check_type({"a": 1, "b": 2}, MutableMapping[str, int])
    assert validator.check_type({"a": 1, "b": 2}, MutableMapping)

    assert validator.check_type({"a": 1, "b": 2}, Mapping[str, int])
    assert validator.check_type({"a": 1, "b": 2}, Mapping)

    assert validator.check_type({"a": 1, "b": 2}, ChainMap[str, int])
    assert validator.check_type({"a": 1, "b": 2}, ChainMap)

    assert validator.check_type({"a": 1, "b": 2}, Counter[str])
    assert validator.check_type({"a": 1, "b": 2}, Counter)

    assert validator.check_type({"a": 1, "b": 2}, DefaultDict[str, int])
    assert validator.check_type({"a": 1, "b": 2}, DefaultDict)

def test_validate_collection_list_types():
    print("test_validate_collection_list_types")
    validator = Validator()
    assert validator.check_type([1, 2], Deque[int])
    assert validator.check_type([1, 2], Deque)

    assert validator.check_type([1, 2], MutableSequence[int])
    assert validator.check_type([1, 2], MutableSequence)

    assert validator.check_type([1, 2], Sequence[int])
    assert validator.check_type([1, 2], Sequence)

    assert not validator.check_type([1, 2], Deque[str])
    assert not validator.check_type([1, 2], MutableSequence[str])
    assert not validator.check_type([1, 2], Sequence[str])

    assert validator.check_type([1, '2'], Sequence[Union[int, str]])

def test_validate_literal_types():
    print("test_validate_literal_types")
    validator = Validator()
    assert validator.check_type(1, Literal[1])
    assert validator.check_type(1, Literal[1, 2])
    assert validator.check_type(1, Literal[1, 2, 3])
    assert validator.check_type(1, Literal[1, 2, 3, 4])
    assert validator.check_type('Good', Literal['Good', 'Bad'])

    assert validator.check_type(1, Literal[1.0, 2.0, 3.0, 4.0])
    assert validator.check_type(1, Literal[1.0, 2.0, 3.0])
    assert validator.check_type(1, Literal[1.0, 2.0])
    assert validator.check_type(1, Literal[1.0])

    assert not validator.check_type(1, Literal[2])
    assert not validator.check_type(1, Literal[2, 3])

    assert not validator.check_type(1, Literal[2.0])
    assert not validator.check_type(1, Literal['Good', 'Bad'])


def test_validate_dataclasses():
    print("test_validate_dataclasses")
    validator = Validator()

    @dataclass
    class Person:
        name: str
        age: int
        height: float
        is_cool: bool
        favourite_numbers: List[int]
        even_more_favourite_numbers: tuple[int, ...]
        favourite_dict: Dict[str, int]

        def __eq__(self, other):
            return self.dict() == other.dict()

        def __hash__(self):
            return hash(str(self.__dict__))

    person = {
        "name": "John",
        "age": 20,
        "height": 1.8,
        "is_cool": True,
        "favourite_numbers": [1, 2, 3],
        "even_more_favourite_numbers": (1, 2, 3),
        "favourite_dict": {"a": 1, "b": 2},
    }
    assert validator.check_type(person, Person)
    assert not validator.check_type(person, str)

    assert validator.check_type([person], List[Person])
    assert not validator.check_type([person], List[str])
    assert validator.check_type([person], List)

    assert validator.check_type([person], Sequence[Person])
    assert not validator.check_type([person], Sequence[str])
    assert validator.check_type([person], Sequence)

    assert validator.check_type({'a': person}, Dict[str, Person])
    assert not validator.check_type({'a': person}, Dict[str, str])
    assert validator.check_type({'a': person}, Dict)

    assert validator.check_type({'a': person}, Mapping[str, Person])
    assert not validator.check_type({'a': person}, Mapping[str, str])
    assert validator.check_type({'a': person}, Mapping)

    assert validator.check_type({'a': person}, MutableMapping[str, Person])
    assert not validator.check_type({'a': person}, MutableMapping[str, str])
    assert validator.check_type({'a': person}, MutableMapping)

    assert validator.check_type({'a': person}, ChainMap[str, Person])
    assert not validator.check_type({'a': person}, ChainMap[str, str])
    assert validator.check_type({'a': person}, ChainMap)

    assert validator.check_type({'a': person}, Counter[str])
    assert not validator.check_type({'a': person}, Counter[int])
    assert validator.check_type({'a': person}, Counter)

    # cant hash dict
    #assert validator.check_type({person}, Set[Person])
    #assert not validator.check_type({person}, Set[str])
    #assert validator.check_type({person}, Set)

def test_validate_pydantic():
    print("test_validate_pydantic")
    validator = Validator()

    class Person(BaseModel):
        name: str
        age: int
        height: float
        is_cool: bool
        favourite_numbers: List[int]
        even_more_favourite_numbers: tuple[int, ...]
        favourite_dict: Dict[str, int]


        def __eq__(self, other):
            return self.model_dump() == other.model_dump()

        def __hash__(self):
            return hash(str(self.model_dump()))

    person = {
        "name": "John",
        "age": 20,
        "height": 1.8,
        "is_cool": True,
        "favourite_numbers": [1, 2, 3],
        "even_more_favourite_numbers": (1, 2, 3),
        "favourite_dict": {"a": 1, "b": 2},
    }
    assert validator.check_type(person, Person)
    assert not validator.check_type(person, str)

    assert validator.check_type([person], List[Person])
    assert not validator.check_type([person], List[str])
    assert validator.check_type([person], List)

    assert validator.check_type([person], Sequence[Person])
    assert not validator.check_type([person], Sequence[str])
    assert validator.check_type([person], Sequence)

    assert validator.check_type({'a': person}, Dict[str, Person])
    assert not validator.check_type({'a': person}, Dict[str, str])
    assert validator.check_type({'a': person}, Dict)

    assert validator.check_type({'a': person}, Mapping[str, Person])
    assert not validator.check_type({'a': person}, Mapping[str, str])
    assert validator.check_type({'a': person}, Mapping)

    assert validator.check_type({'a': person}, MutableMapping[str, Person])
    assert not validator.check_type({'a': person}, MutableMapping[str, str])
    assert validator.check_type({'a': person}, MutableMapping)

    assert validator.check_type({'a': person}, ChainMap[str, Person])
    assert not validator.check_type({'a': person}, ChainMap[str, str])
    assert validator.check_type({'a': person}, ChainMap)

    assert validator.check_type({'a': person}, Counter[str])
    assert not validator.check_type({'a': person}, Counter[int])
    assert validator.check_type({'a': person}, Counter)

    # cant hash dict
    #assert validator.check_type({person}, Set[Person])
    #assert not validator.check_type({person}, Set[str])
    #assert validator.check_type({person}, Set)


if __name__ == "__main__":
    test_validate_pydantic()
    #test_validate_dataclasses()
    #test_validate_literal_types()
    #test_validate_collection_list_types()
    #test_validate_collection_dict_types()
    #test_validate_type_annotations()
    #test_validate_complex_types()
    #test_validate_base_types()