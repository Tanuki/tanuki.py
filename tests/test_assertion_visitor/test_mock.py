import ast
import datetime
import pprint

from typing import Optional, List

from pydantic import BaseModel

import tanuki
from tanuki.assertion_visitor import AssertionVisitor, Or


def _parse(source):
    class TodoItem(BaseModel):
        deadline: Optional[datetime.datetime] = None
        goal: str
        people: List[str]

    @tanuki.patch
    def create_todolist_item(goal: str, people=[]) -> TodoItem:
        pass

    tree = ast.parse(source)
    _locals = locals()
    visitor = AssertionVisitor(locals(), patch_symbolic_funcs={"create_todolist_item": create_todolist_item})
    visitor.visit(tree)
    return visitor.mocks

def test_rl_equality():
    source = \
""" 
assert TodoItem(goal="Go to the store and buy some milk", people=["Me"]) == create_todolist_item("I would like to go to the shop and buy some milk")
"""

    mocks = _parse(source)

    assert len(mocks) == 1

def test_equality():

    source = \
""" 
assert create_todolist_item("I would like to go to the shop and buy some milk") == TodoItem(goal="Go to the store and buy some milk", people=["Me"])
"""

    mocks = _parse(source)

    assert len(mocks) == 1


def test_non_equality():
    """
    Should not create a mock, because we need input - output pairs.
    """
    source = \
"""
assert create_todolist_item("I need to go and visit George at 3pm tomorrow") != TodoItem(goal="Go and visit George", people=["Me"], deadline=datetime(2021, 1, 1, 14, 0))
"""

    mocks = _parse(source)

    assert len(mocks) == 0


def test_none():
    source = \
"""
assert create_todolist_item("Invalid input") is None
"""
    mocks = _parse(source)

    assert len(mocks) == 1

def test_not_none():
    """
    This should also not result in a mock, because nEq is insufficient to map input to output.
    :return:
    """
    source = \
"""
assert create_todolist_item("I would like to go to the shop and buy some tomatoes") is not None
"""
    mocks = _parse(source)

    assert len(mocks) == 0


def test_inclusion():
    """
    TODO: Review whether it is a good idea to output a list of valid outputs. This may occlude legitimate list outputs.
    Perhaps we want to create a special list-like object that can be used to represent a list of valid outputs.
    :return:
    """
    source = \
"""
import datetime
valid_outputs = [
        TodoItem(goal="Go to the store and buy some milk", people=["Me"]),
        TodoItem(goal="Go and visit Jeff", people=["Me"], deadline=datetime.datetime(2021, 1, 1, 15, 0))
    ]
assert create_todolist_item("I would like to go to the shop and buy some cheese") in valid_outputs
"""
    mocks = _parse(source)

    assert len(list(mocks.values())[0]) == 2


def test_exclusion():
    source = \
"""
import datetime
invalid_outputs = [
        TodoItem(goal="Go to the gym", people=["Me"]),
        TodoItem(goal="Go and visit Sarah", people=["Me"], deadline=datetime(2021, 1, 1, 16, 0))
    ]
assert create_todolist_item("I would like to go to the terrace and buy some milk") not in invalid_outputs
"""
    mocks = _parse(source)

    assert len(mocks) == 0


def test_tuple_iteration():
    source = \
"""
import datetime
inputs = ["I would like to go to the store and buy some milk", "I need to go and visit Jeff at 3pm tomorrow"]
outputs = [TodoItem(goal="Go to the store and buy some milk", people=["Me"]), TodoItem(goal="Go and visit Jeff", people=["Me"], deadline=datetime(2021, 1, 1, 15, 0))]
for input, output in zip(inputs, outputs):
    assert create_todolist_item(input) == output
"""
    mocks = _parse(source)

    assert len(mocks) == 2

def test_iteration():
    source = \
"""
import datetime
inputs = ["I would like to go to the store and buy some milk", "I need to go and visit Jeff at 3pm tomorrow"]
outputs = [TodoItem(goal="Go to the store and buy some milk", people=["Me"]), TodoItem(goal="Go and visit Jeff", people=["Me"], deadline=datetime(2021, 1, 1, 15, 0))]
for input in inputs:
    assert create_todolist_item(input) in outputs
"""
    mocks = _parse(source)

    assert len(mocks) == 2

    assert len(list(mocks.values())[0]) == 2
    assert len(list(mocks.values())[1]) == 2

    assert isinstance(list(mocks.values())[0], Or)

if __name__ == "__main__":
    test_equality()
    test_iteration()
    test_non_equality()
    test_none()
    test_not_none()
    test_inclusion()
    test_exclusion()
    test_tuple_iteration()
    test_rl_equality()

