import os
import unittest
from typing import Literal, Optional, List, Union
from pydantic import BaseModel, Field
import openai
from dotenv import load_dotenv

import tanuki

load_dotenv()
from tanuki.register import Register
openai.api_key = os.getenv("OPENAI_API_KEY")

class Person(BaseModel):
    age: int = Field(..., ge=0, le=155)
    name: str
    favourite_colours: List[str]



@tanuki.patch
def summarise_list_generic(input: List[str]) -> str:
    """
    Summarise the input
    """
@tanuki.patch
def summarise_list_typing(input: list[str]) -> str:
    """
    Summarise the input
    """
@tanuki.patch
def summarise_list_pydantic(input: List[Person]) -> str:
    """
    Summarise the input
    """
@tanuki.patch
def summarise_list_dict(input: List[dict]) -> str:
    """
    Summarise the input
    """

@tanuki.patch
def summarise_list_int(input: List[int]) -> str:
    """
    Summarise the input
    """

@tanuki.patch
def summarise_list_Union(input: List[Union[int, float]]) -> str:
    """
    Summarise the input
    """

@tanuki.align
def align_list_generic():
    assert summarise_list_generic(["Thats awesome", "Thats cool"]) == 'They found it awesome and cool'
    assert summarise_list_generic(["Thats neat", "Thats ok"]) == 'They found it neat and ok'

    assert summarise_list_generic(input = ["Thats awesome", "Thats cool"]) == 'They found it awesome and cool'
    assert summarise_list_generic(input = ["Thats neat", "Thats ok"]) == 'They found it neat and ok'

@tanuki.align
def align_list_typing():
    assert summarise_list_typing(["Thats awesome", "Thats cool"]) == 'They found it awesome and cool'
    assert summarise_list_typing(["Thats neat", "Thats ok"]) == 'They found it neat and ok'

    assert summarise_list_typing(input = ["Thats awesome", "Thats cool"]) == 'They found it awesome and cool'
    assert summarise_list_typing(input = ["Thats neat", "Thats ok"]) == 'They found it neat and ok'


@tanuki.align
def align_list_pydantic():
    person_1 = Person(name="Jeff", age=25, favourite_colours=["Red", "Blue"])
    person_2 = Person(name="Thomas", age=33, favourite_colours=["Green", "Gray"])
    input = [person_1, person_2]
    assert summarise_list_pydantic(input) == "Jeff is 25 years old and likes Red and Blue. Thomas is 33 years old and likes Green and Gray."
    assert summarise_list_pydantic(input = input) == "Jeff is 25 years old and likes Red and Blue. Thomas is 33 years old and likes Green and Gray."


@tanuki.align
def align_list_dict():
    input = [{"name": "Jeff", "age": 25, "favourite_colours": ["Red", "Blue"]}, {"name": "Thomas", "age": 33, "favourite_colours": ["Green", "Gray"]}]
    assert summarise_list_dict(input) == "Jeff is 25 years old and likes Red and Blue. Thomas is 33 years old and likes Green and Gray."
    assert summarise_list_dict(input = input) == "Jeff is 25 years old and likes Red and Blue. Thomas is 33 years old and likes Green and Gray."


@tanuki.align
def align_list_int():
    assert summarise_list_int([1, 2]) == "The sum of the list is 3"
    assert summarise_list_int([1, 2, 3]) == "The sum of the list is 6"

    assert summarise_list_int(input = [1, 2]) == "The sum of the list is 3"
    assert summarise_list_int(input = [1, 2, 3]) == "The sum of the list is 6"

@tanuki.align
def align_list_Union():
    assert summarise_list_Union([1, 2]) == "The sum of the list is 3"
    assert summarise_list_Union([1.0, 2.0, 3.0]) == "The sum of the list is 6.0"

    assert summarise_list_Union(input = [1, 2]) == "The sum of the list is 3"
    assert summarise_list_Union(input = [1.0, 2.0, 3.0]) == "The sum of the list is 6.0"

def test_list():
    # This tests all the list aligns
    # can be called by pytest or unittest
    align_list_generic()
    align_list_typing()
    align_list_pydantic()
    align_list_dict()
    align_list_int()
    align_list_Union()
    print("All list aligns passed!")


@tanuki.patch
def summarise_str(input: str) -> str:
    """
    Summarise the input
    """

@tanuki.patch
def summarise_pydantic(input: Person) -> str:
    """
    Summarise the input
    """
@tanuki.patch
def summarise_dict(input: dict) -> str:
    """
    Summarise the input
    """

@tanuki.patch
def summarise_int(input: int) -> str:
    """
    Summarise the input
    """

@tanuki.patch
def summarise_Union(input: Union[int, float]) -> str:
    """
    Summarise the input
    """

@tanuki.align
def align_string():
    assert summarise_str("Thats awesome") == 'They found it awesome and cool'
    assert summarise_str("Thats neat") == 'They found it neat and ok'

    assert summarise_str(input = "Thats awesome") == 'They found it awesome and cool'
    assert summarise_str(input = "Thats neat") == 'They found it neat and ok'


@tanuki.align
def align_pydantic():
    person_1 = Person(name="Jeff", age=25, favourite_colours=["Red", "Blue"])
    input = person_1
    assert summarise_pydantic(input) == "Jeff is 25 years old and likes Red and Blue."

    assert summarise_pydantic(input = input) == "Jeff is 25 years old and likes Red and Blue."

@tanuki.align
def align_dict():
    input = {"name": "Jeff", "age": 25, "favourite_colours": ["Red", "Blue"]}
    assert summarise_dict(input) == "Jeff is 25 years old and likes Red and Blue."

    assert summarise_dict(input = input) == "Jeff is 25 years old and likes Red and Blue."


@tanuki.align
def align_list_int():
    assert summarise_int(1) == "This is number 1"
    assert summarise_int(input = 1) == "This is number 1"

@tanuki.align
def align_list_Union():
    assert summarise_Union(1) == "This is number 1"
    assert summarise_Union(2.0) == "This is number 2"

    assert summarise_Union(input = 1) == "This is number 1"
    assert summarise_Union(input = 2.0) == "This is number 2"

def test_single():
    # This tests all the single aligns
    # can be called by pytest or unittest
    align_string()
    align_pydantic()
    align_dict()
    align_list_int()
    align_list_Union()
    print("All single aligns passed!")

def _parse_examples(test_func):
    # check that all the examples are correctly readable
    function_description = Register.load_function_description(test_func)
    function_modeler = tanuki.function_modeler
    align_examples = function_modeler.get_symbolic_alignments(function_description.__hash__())
    examples = "\n".join([f"Inputs:\nArgs: {align['args']}\nKwargs: {align['kwargs']}\nOutput: {align['output']}" for align in align_examples])

def test_parse_align_datasets():
    # Test that all the examples that are aligned are correctly parsable into the prompt format we have defined
    _parse_examples(summarise_list_generic)
    _parse_examples(summarise_list_typing)
    _parse_examples(summarise_list_pydantic)
    _parse_examples(summarise_list_dict)
    _parse_examples(summarise_list_int)
    _parse_examples(summarise_list_Union)
    _parse_examples(summarise_str)
    _parse_examples(summarise_pydantic)
    _parse_examples(summarise_dict)
    _parse_examples(summarise_int)
    _parse_examples(summarise_Union)
    print("All examples parsed correctly!")

