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
def summarise_list_generic(input: str) -> List[str]:
    """
    Summarise the input into multiple sentences in a list
    """
@tanuki.patch
def summarise_list_typing(input: str) -> list[str]:
    """
    Summarise the input into multiple sentences in a list
    """
@tanuki.patch
def summarise_list_pydantic(input: str) -> List[Person]:
    """
    Create a list of Personas
    """
@tanuki.patch
def summarise_list_dict(input: str) -> List[dict]:
    """
    Create a list of dict personas
    """

@tanuki.patch
def summarise_list_int(input: str) -> List[int]:
    """
    Extract the integers
    """

@tanuki.patch
def summarise_list_Union(input: str) -> List[Union[int, float]]:
    """
    Extract the numbers
    """

@tanuki.align
def align_list_generic():

    assert summarise_list_generic("Thats awesome. Thats cool") == ["Thats awesome", "Thats cool"]
    assert summarise_list_generic("Thats neat. Thats ok") == ["Thats neat", "Thats ok"]

    assert summarise_list_generic(input = "Thats awesome. Thats cool") == ["Thats awesome", "Thats cool"]
    assert summarise_list_generic(input = "Thats neat. Thats ok") == ["Thats neat", "Thats ok"]

@tanuki.align
def align_list_typing():
    assert summarise_list_typing("Thats awesome. Thats cool") == ["Thats awesome", "Thats cool"]
    assert summarise_list_typing("Thats neat. Thats ok") == ["Thats neat", "Thats ok"]

    assert summarise_list_typing(input = "Thats awesome. Thats cool") == ["Thats awesome", "Thats cool"]
    assert summarise_list_typing(input = "Thats neat. Thats ok") == ["Thats neat", "Thats ok"]


@tanuki.align
def align_list_pydantic():

    person_str = "First person - Name Jeff, age 25, favourite colours Red and Blue. Second person - Name Thomas, age 33, favourite colours Green and Gray"
    output = [Person(name="Jeff", age=25, favourite_colours=["Red", "Blue"]), Person(name="Thomas", age=33, favourite_colours=["Green", "Gray"])]
    assert summarise_list_pydantic(person_str) == output
    assert summarise_list_pydantic(input = person_str) == output


@tanuki.align
def align_list_dict():
    person_str = "First person - Name Jeff, age 25, favourite colours Red and Blue. Second person - Name Thomas, age 33, favourite colours Green and Gray"
    output = [{"name": "Jeff", "age": 25, "favourite_colours": ["Red", "Blue"]}, {"name": "Thomas", "age": 33, "favourite_colours": ["Green", "Gray"]}]
    assert summarise_list_dict(person_str) == output
    assert summarise_list_dict(input = person_str) == output


@tanuki.align
def align_list_int():

    input_1 = "1 and 2"
    input_2 = "1, 2 and 3"

    assert summarise_list_int(input_1) == [1, 2]
    assert summarise_list_int(input_2) == [1, 2, 3]

    assert summarise_list_int(input = input_1) == [1, 2]
    assert summarise_list_int(input = input_2) == [1, 2, 3]

@tanuki.align
def align_list_Union():
    input_1 = "1 and 2"
    input_2 = "1.0, 2.0 and 3.0"

    assert summarise_list_Union(input_1) == [1, 2]
    assert summarise_list_Union(input_2) ==[1.0, 2.0, 3.0]

    assert summarise_list_Union(input = input_1) == [1, 2]
    assert summarise_list_Union(input = input_2) == [1.0, 2.0, 3.0]

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
    Summarise the input into 1 sentence
    """

@tanuki.patch
def summarise_pydantic(input: str) -> Person:
    """
    Create the persona
    """
@tanuki.patch
def summarise_dict(input: str) -> dict:
    """
    Create the persona
    """

@tanuki.patch
def summarise_int(input: str) -> int:
    """
    Extract the integer
    """

@tanuki.patch
def summarise_Union(input: str) -> Union[int, float]:
    """
    Extract the number
    """

@tanuki.align
def align_string():
    assert summarise_str("Thats awesome. Thats cool") == 'They found it awesome and cool'
    assert summarise_str("Thats neat. Thats ok") == 'They found it neat and ok'

    assert summarise_str(input = "Thats awesome. Thats cool") == 'They found it awesome and cool'
    assert summarise_str(input = "Thats neat. Thats OK") == 'They found it neat and ok'


@tanuki.align
def align_pydantic():
    input_str = "Name Jeff, age 25, favourite colours Red and Blue"
    person = Person(name="Jeff", age=25, favourite_colours=["Red", "Blue"])
    assert summarise_pydantic(input_str) == person

    assert summarise_pydantic(input = input_str) == person

@tanuki.align
def align_dict():
    input_str = "Name Jeff, age 25, favourite colours Red and Blue"
    output = {"name": "Jeff", "age": 25, "favourite_colours": ["Red", "Blue"]}
    assert summarise_dict(input_str) == output

    assert summarise_dict(input = input_str) == output


@tanuki.align
def align_list_int():
    input_str = "This is number 1"
    assert summarise_int(input_str) == 1
    assert summarise_int(input = input_str) == 1

@tanuki.align
def align_list_Union():
    input_str_1 = "This is number 1"
    input_str_2 = "This is number 2.0"
    assert summarise_Union(input_str_1) == 1
    assert summarise_Union(input_str_2) == 2.0

    assert summarise_Union(input = input_str_1) == 1
    assert summarise_Union(input = input_str_2) == 2.0

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
