from collections import deque, defaultdict
from dataclasses import dataclass
from typing import List, Dict, Tuple, Union, Optional, Sequence, Iterable, Mapping, ChainMap, Set, get_args, FrozenSet, \
    DefaultDict, Counter, OrderedDict

import pytest
from pydantic import BaseModel

from tanuki.validator import Validator


def test_instantiate_pydantic():
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

    validator = Validator()
    person = {
        "name": "John",
        "age": 20,
        "height": 1.8,
        "is_cool": True,
        "favourite_numbers": [1, 2, 3],
        "even_more_favourite_numbers": (1, 2, 3),
        "favourite_dict": {"a": 1, "b": 2},
    }
    person_obj = validator.instantiate(person, Person)
    assert isinstance(person_obj, Person)
    # test lists
    list_pydantic = [person, person]
    person_obj = validator.instantiate(list_pydantic, List[Person])
    assert isinstance(person_obj, list)
    assert isinstance(person_obj[0], Person)
    assert isinstance(person_obj[1], Person)
    assert len(person_obj) == 2

    # Nested data classes or Pydantic models.
    @dataclass
    class InnerDataClass:
        y: int

    @dataclass
    class OuterDataClass:
        x: InnerDataClass

    data = {"x": {"y": 10}}
    assert validator.instantiate(data, OuterDataClass) == OuterDataClass(InnerDataClass(10))

def test_instantiate_dataclass():
    validator = Validator()
    @dataclass
    class Person:
        name: str
        age: int
        height: float
        is_cool: bool

    person = {
        "name": "John",
        "age": 20,
        "height": 1.8,
        "is_cool": True
    }
    person_obj = validator.instantiate(person, Person)
    assert isinstance(person_obj, Person)

def test_primitives():
    """
    Test that primitives and empty objects are instantiated correctly.
    """
    validator = Validator()
    assert validator.instantiate("1", int) == 1
    assert validator.instantiate("1.0", float) == 1.0
    assert validator.instantiate("true", bool) == True
    assert isinstance(validator.instantiate("1", float), float)
    assert isinstance(validator.instantiate("1.0", int), int)
    assert validator.instantiate("1", str) != 1
    assert validator.instantiate("1.0", str) != 1.0
    assert validator.instantiate("true", str) != True
    assert validator.instantiate({}, dict) == {}
    assert validator.instantiate({"asd": 2, "bb": "ad"}, dict) == {"asd": 2, "bb": "ad"}
    assert validator.instantiate([], list) == []
    assert validator.instantiate((), tuple) == ()
    assert validator.instantiate((1,2), tuple) == (1, 2)
    assert validator.instantiate(set(), frozenset) == set()
    assert validator.instantiate((), frozenset) == ()
    assert validator.instantiate((), set) == ()

def test_generics():
    """
    Test that generics are instantiated correctly.
    """
    validator = Validator()

    # Lists, dictionaries, tuples, sets, and frozensets.
    assert isinstance(validator.instantiate([1, 2, 3], List[int]), List)
    assert validator.instantiate({"a": 1, "b": 2}, Dict[str, int]) == {"a": 1, "b": 2}
    assert validator.instantiate((1, "a", 2.0), Tuple[int, str, float]) == (1, "a", 2.0)

    # Lists of lists, dictionaries of dictionaries, etc.
    assert validator.instantiate([[1, 2], [3, 4]], List[List[int]]) == [[1, 2], [3, 4]]
    assert validator.instantiate({"a": {"b": 2}, "c": {"d": 4}}, Dict[str, Dict[str, int]]) == {"a": {"b": 2},
                                                                                                "c": {"d": 4}}
    # Lists that can contain multiple types of items.
    assert validator.instantiate([1, "a", 2.0], List[Union[int, str, float]]) == [1, "a", 2.0]
    assert validator.instantiate({"a": 1, "b": "c"}, Dict[str, Union[int, str]]) == {"a": 1, "b": "c"}

    # Items that can be None.
    assert validator.instantiate([1, None, 2], List[Optional[int]]) == [1, None, 2]

    # Combinations of the above.
    data = {"a": [1, 2, 3], "b": [4, "c", 6]}
    assert validator.instantiate(data, Dict[str, List[Union[int, str]]]) == data

    # Data that doesn't match the provided type should raise an error.
    try:
        validator.instantiate([1, 2, "a"], List[int])
        assert False, "Expected a TypeError"
    except TypeError:
        pass

    # Ensure that custom classes without generics work as expected.
    class SimpleClass:
        def __init__(self, x):
            self.x = x

    assert isinstance(validator.instantiate({"x": 10}, SimpleClass), SimpleClass)

    # ABCs like Sequence, Iterable, Mapping instead of concrete types like list, tuple, dict.
    assert validator.instantiate([1, 2, 3], Sequence[int]) == [1, 2, 3]
    assert validator.instantiate((1, 2, 3), Iterable[int]) == (1, 2, 3)
    assert validator.instantiate({"a": 1, "b": 2}, Mapping[str, int]) == {"a": 1, "b": 2}

    # Deque
    result = validator.instantiate([1, 2, 3], deque)
    assert isinstance(result, deque) and list(result) == [1, 2, 3]

    # FrozenSet
    result = validator.instantiate([1, 2, 3], FrozenSet[int])
    assert isinstance(result, frozenset) and list(result) == [1, 2, 3]

    # DefaultDict
    result = validator.instantiate({"a": 1, "b": 2}, DefaultDict[str, int])
    assert isinstance(result, defaultdict) and dict(result) == {"a": 1, "b": 2}

    # Counter
    #result = validator.instantiate({"a": 1, "b": 2}, Counter[str])
    #assert isinstance(result, Counter) and dict(result) == {"a": 1, "b": 2}

    # OrderedDict
    result = validator.instantiate({"a": 1, "b": 2}, OrderedDict[str, int])
    assert isinstance(result, OrderedDict) and dict(result) == {"a": 1, "b": 2}

@pytest.fixture
def validator():
    return Validator()

def test_extended_generics(validator):

    test_extended_list(validator)
    test_extended_tuple(validator)
    test_extended_dict(validator)
    test_extended_set(validator)


def test_extended_set(validator):
    class ExtendedSet(Set[int]):
        pass

    # We are using a list here because that is how they are represented in JSON.
    extended_set = validator.instantiate([1, 2, 3], ExtendedSet)
    assert isinstance(extended_set, ExtendedSet)
    # Test type coercion
    extended_set = validator.instantiate([1, "2", 3], ExtendedSet)
    assert isinstance(extended_set, ExtendedSet)
    # Lets now try and violate the inherited type
    try:
        validator.instantiate([1, "a", 3], ExtendedSet)
        assert False, "Expected a ValueError"
    except TypeError:
        pass


def test_extended_dict(validator):
    class ExtendedDict(Dict[str, int]):
        pass

    extended_dict = validator.instantiate({"a": 1, "b": 2}, ExtendedDict)
    assert isinstance(extended_dict, ExtendedDict)
    # Test type coercion
    extended_dict = validator.instantiate({"a": "1", "b": 2}, ExtendedDict)
    assert isinstance(extended_dict, ExtendedDict)
    # Lets now try and violate the inherited type
    try:
        validator.instantiate({4: 1, "b": 2}, ExtendedDict)
        assert False, "Expected a ValueError"
    except TypeError:
        pass


def test_extended_tuple(validator):
    class ExtendedTuple(Tuple[int, str]):
        pass

    # We are using a list here because that is how they are represented in JSON.
    extended_tuple = validator.instantiate([1, "a"], ExtendedTuple)
    assert isinstance(extended_tuple, ExtendedTuple)
    # Test type coercion
    extended_tuple = validator.instantiate(["1", 1], ExtendedTuple)
    assert isinstance(extended_tuple, ExtendedTuple)
    # Lets now try and violate the inherited type
    try:
        validator.instantiate(['a', 2], ExtendedTuple)
        assert False, "Expected a ValueError"
    except TypeError:
        pass


def test_extended_list(validator):
    class ExtendedList(List[int]):
        pass

    extended_list = validator.instantiate([1, 2, 3], ExtendedList)
    assert isinstance(extended_list, ExtendedList)
    # Lets now try and violate the inherited type
    try:
        validator.instantiate([1, "a"], ExtendedList)
        assert False, "Expected a ValueError"
    except TypeError:
        pass


if __name__ == "__main__":
    test_instantiate_pydantic()
    test_instantiate_dataclass()
    test_primitives()
    test_generics()
    test_extended_generics(Validator())